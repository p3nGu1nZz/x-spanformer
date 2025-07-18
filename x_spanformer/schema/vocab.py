from pydantic import BaseModel, Field, ConfigDict


class VocabPiece(BaseModel):
    """
    Represents a vocabulary piece from the Unigram-LM induction process.
    
    Following Section 3.1 of the X-Spanformer paper, each piece u ∈ V has:
    - piece: the substring u itself
    - prob: p(u), the learned piece probability from EM
    """
    piece: str = Field(..., description="The vocabulary piece (substring)")
    prob: float = Field(..., gt=0.0, le=1.0, description="Piece probability p(u) from Unigram-LM")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "piece": "the",
                "prob": 0.01234
            }
        }
    )


class VocabStats(BaseModel):
    """
    Statistics about the induced vocabulary following the paper's metrics.
    
    Following Section 3.1 of the X-Spanformer paper, this captures:
    - Final vocabulary size |V|
    - Perplexity metrics before and after pruning
    - OOV rate (uncovered codepoint positions)
    - EM algorithm statistics
    """
    total_pieces: int = Field(..., description="Final vocabulary size |V|")
    baseline_ppl: float = Field(..., description="Initial corpus perplexity PPL^(0)")
    final_ppl: float = Field(..., description="Final corpus perplexity after pruning")
    oov_rate: float = Field(..., ge=0.0, le=1.0, description="Out-of-vocabulary rate (uncovered positions)")
    em_iterations: int = Field(..., description="Number of EM iterations performed")
    pruned_pieces: int = Field(default=0, description="Number of pieces pruned during adaptation")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_pieces": 8192,
                "baseline_ppl": 245.7,
                "final_ppl": 251.3,
                "oov_rate": 0.001,
                "em_iterations": 5,
                "pruned_pieces": 1234
            }
        }
    )
