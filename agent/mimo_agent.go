package agent

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
)

// MiMoAgent invokes MiMo Code CLI via streaming.
type MiMoAgent struct {
	name         string
	command      string
	args         []string
	cwd          string
	env          map[string]string
	model        string
	systemPrompt string
	mu           sync.Mutex
	sessions     map[string]string
	sessionsPath string // path to sessions.json for persistence
}

// MiMoAgentConfig holds configuration for a MiMo agent.
type MiMoAgentConfig struct {
	Name         string
	Command      string
	Args         []string
	Cwd          string
	Env          map[string]string
	Model        string
	SystemPrompt string
}

// sessionsFilePath returns the path to the sessions persistence file.
func sessionsFilePath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".weclaw", "sessions.json"), nil
}

// loadSessions loads sessions from disk.
func loadSessions(path string) map[string]string {
	sessions := make(map[string]string)
	if path == "" {
		return sessions
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return sessions
		}
		log.Printf("[mimo] warning: failed to load sessions: %v", err)
		return sessions
	}
	if err := json.Unmarshal(data, &sessions); err != nil {
		log.Printf("[mimo] warning: failed to parse sessions: %v", err)
		return sessions
	}
	log.Printf("[mimo] loaded %d sessions from disk", len(sessions))
	return sessions
}

// saveSessions saves sessions to disk.
func saveSessions(path string, sessions map[string]string) {
	if path == "" {
		return
	}
	data, err := json.MarshalIndent(sessions, "", "  ")
	if err != nil {
		log.Printf("[mimo] warning: failed to marshal sessions: %v", err)
		return
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		log.Printf("[mimo] warning: failed to save sessions: %v", err)
	}
}

// NewMiMoAgent creates a new MiMo agent.
func NewMiMoAgent(cfg MiMoAgentConfig) *MiMoAgent {
	cwd := cfg.Cwd
	if cwd == "" {
		cwd = defaultWorkspace()
	}
	sessionsPath, _ := sessionsFilePath()
	return &MiMoAgent{
		name:         cfg.Name,
		command:      cfg.Command,
		args:         cfg.Args,
		cwd:          cwd,
		env:          cfg.Env,
		model:        cfg.Model,
		systemPrompt: cfg.SystemPrompt,
		sessions:     loadSessions(sessionsPath),
		sessionsPath: sessionsPath,
	}
}

// Info returns metadata about this agent.
func (a *MiMoAgent) Info() AgentInfo {
	return AgentInfo{
		Name:    a.name,
		Type:    "mimo",
		Model:   a.model,
		Command: a.command,
	}
}

// ResetSession clears the existing session for the given conversationID.
func (a *MiMoAgent) ResetSession(_ context.Context, conversationID string) (string, error) {
	a.mu.Lock()
	delete(a.sessions, conversationID)
	saveSessions(a.sessionsPath, a.sessions)
	a.mu.Unlock()
	log.Printf("[mimo] session reset (conversation=%s)", conversationID)
	return "", nil
}

// SetCwd changes the working directory for subsequent operations.
func (a *MiMoAgent) SetCwd(cwd string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.cwd = cwd
}

// mimoEvent represents a single JSON event from `mimo run --format json`.
type mimoEvent struct {
	Type      string `json:"type"`
	SessionID string `json:"sessionID"`
	Part      *struct {
		Text string `json:"text"`
	} `json:"part,omitempty"`
}

// Chat sends a message to the MiMo agent and returns the response.
func (a *MiMoAgent) Chat(ctx context.Context, conversationID string, message string) (string, error) {
	args := []string{"run", message, "--dangerously-skip-permissions", "--format", "json"}

	if a.model != "" {
		args = append(args, "--model", a.model)
	}
	if a.systemPrompt != "" {
		args = append(args, "--title", a.systemPrompt)
	}
	args = append(args, a.args...)

	// Check for existing session
	a.mu.Lock()
	sessionID, hasSession := a.sessions[conversationID]
	a.mu.Unlock()

	if hasSession {
		args = append(args, "--session", sessionID)
		log.Printf("[mimo] resuming session (session=%s, conversation=%s)", sessionID, conversationID)
	} else {
		log.Printf("[mimo] starting new conversation (conversation=%s)", conversationID)
	}

	cmd := exec.CommandContext(ctx, a.command, args...)
	if a.cwd != "" {
		cmd.Dir = a.cwd
	}
	if len(a.env) > 0 {
		cmdEnv, err := mergeEnv(os.Environ(), a.env)
		if err != nil {
			return "", fmt.Errorf("build %s env: %w", a.name, err)
		}
		cmd.Env = cmdEnv
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return "", fmt.Errorf("create stdout pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("start %s: %w", a.name, err)
	}

	// Parse streaming JSON events to extract sessionID and text
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 0, 1024*1024), 1024*1024)

	var texts []string
	var newSessionID string

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		var event mimoEvent
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			continue
		}
		if event.SessionID != "" {
			newSessionID = event.SessionID
		}
		if event.Type == "text" && event.Part != nil && event.Part.Text != "" {
			texts = append(texts, event.Part.Text)
		}
	}

	if err := cmd.Wait(); err != nil {
		return "", fmt.Errorf("%s error: %w", a.name, err)
	}

	log.Printf("[mimo] process exited (command=%s)", a.command)

	// Save session ID for multi-turn conversation
	if newSessionID != "" {
		a.mu.Lock()
		a.sessions[conversationID] = newSessionID
		saveSessions(a.sessionsPath, a.sessions)
		a.mu.Unlock()
		log.Printf("[mimo] saved session (session=%s, conversation=%s)", newSessionID, conversationID)
	}

	response := strings.TrimSpace(strings.Join(texts, ""))
	if response == "" {
		return "", fmt.Errorf("%s returned empty response", a.name)
	}

	return response, nil
}
