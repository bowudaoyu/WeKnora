package vlm

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestQwenNoThinkingTransport(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		require.NoError(t, json.Unmarshal(raw, &got))
		assert.EqualValues(t, len(raw), r.ContentLength)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	client := &http.Client{Transport: qwenNoThinkingTransport{base: http.DefaultTransport}}
	resp, err := client.Post(srv.URL, "application/json",
		strings.NewReader(`{"model":"qwen-vl-max","max_tokens":5000,"temperature":0.1}`))
	require.NoError(t, err)
	resp.Body.Close()

	assert.Equal(t, false, got["enable_thinking"])
	assert.Equal(t, false, got["enable_search"])
	assert.Equal(t, "qwen-vl-max", got["model"])
	assert.EqualValues(t, 5000, got["max_tokens"])
}

func TestRemoteAPIVLMSelfHostedQwenDisablesThinking(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		require.NoError(t, json.Unmarshal(raw, &got))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	defer srv.Close()

	v, err := NewRemoteAPIVLM(&Config{
		Provider:  "generic",
		BaseURL:   srv.URL,
		ModelName: "qwen3.8-27b-local",
	})
	require.NoError(t, err)
	_, err = v.Predict(t.Context(), [][]byte{[]byte("\x89PNG\r\n\x1a\n")}, "describe")
	require.NoError(t, err)

	assert.Equal(t, map[string]any{"enable_thinking": false}, got["chat_template_kwargs"])
	assert.NotContains(t, got, "enable_search")
}
