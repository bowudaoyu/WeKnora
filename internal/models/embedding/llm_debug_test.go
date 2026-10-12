package embedding

import (
	"context"
	"testing"
)

type poolRecorder struct {
	Embedder
	gotModel Embedder
}

func (p *poolRecorder) BatchEmbedWithPool(_ context.Context, model Embedder, _ []string) ([][]float32, error) {
	p.gotModel = model
	return nil, nil
}

// The debug wrapper sits below the Langfuse wrapper; it must hand the pool the
// outer embedder it was given, otherwise pooled batches bypass Langfuse.
func TestDebugEmbedderKeepsOuterModelForPool(t *testing.T) {
	rec := &poolRecorder{}
	d := &debugEmbedder{inner: rec}
	outer := &langfuseEmbedder{inner: d}

	if _, err := d.BatchEmbedWithPool(context.Background(), outer, []string{"x"}); err != nil {
		t.Fatal(err)
	}
	if rec.gotModel != Embedder(outer) {
		t.Fatalf("pool got %T; want the outer langfuse embedder", rec.gotModel)
	}

	if _, err := d.BatchEmbedWithPool(context.Background(), nil, []string{"x"}); err != nil {
		t.Fatal(err)
	}
	if rec.gotModel != Embedder(d) {
		t.Fatalf("pool got %T; want the debug embedder itself when model is nil", rec.gotModel)
	}
}
