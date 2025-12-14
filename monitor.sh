#!/bin/bash
# Monitor QuantGAN training progress

echo "========================================"
echo "  QuantGAN Training Monitor"
echo "========================================"
echo ""

# Check if process is running
PID=$(pgrep -f "train_ohlcv" | head -1)

if [ -z "$PID" ]; then
    echo "❌ No training process found!"
    echo ""
    echo "Check completed checkpoints:"
    ls -la checkpoints/*/netG_best.pth 2>/dev/null | tail -5
    exit 1
fi

echo "✅ Training is RUNNING (PID: $PID)"
echo ""

# Get process stats
echo "📊 Resource Usage:"
ps -p $PID -o pcpu,pmem,etime | tail -1 | awk '{print "   CPU: "$1"% | Memory: "$2"% | Elapsed: "$3}'
echo ""

# Check latest checkpoint folder
LATEST_CKPT=$(ls -td checkpoints/BTCUSDT_5m_tcn_* 2>/dev/null | head -1)

if [ -n "$LATEST_CKPT" ]; then
    echo "📁 Checkpoint folder: $LATEST_CKPT"
    
    # Count saved models
    N_CKPTS=$(ls "$LATEST_CKPT"/*.pth 2>/dev/null | wc -l | tr -d ' ')
    echo "   Saved checkpoints: $N_CKPTS"
    
    # Check training.log for completed epochs
    if [ -f "$LATEST_CKPT/training.log" ]; then
        EPOCHS=$(grep -c "^Epoch" "$LATEST_CKPT/training.log" 2>/dev/null)
        EPOCHS=${EPOCHS:-0}  # Default to 0 if empty
        EPOCHS=$(echo "$EPOCHS" | tr -d '[:space:]')  # Remove whitespace
        echo "   Completed epochs: $EPOCHS / 100"
        
        if [ "$EPOCHS" -gt 0 ] 2>/dev/null; then
            echo ""
            echo "📈 Latest progress:"
            tail -3 "$LATEST_CKPT/training.log"
        fi
    else
        echo "   Completed epochs: 0 / 100 (first epoch in progress)"
    fi
fi

echo ""
echo "📝 Live log (last 5 lines):"
tail -5 training_full.log 2>/dev/null || echo "   (no output yet)"

echo ""
echo "========================================"
echo "Run again: ./monitor.sh"
echo "Live tail: tail -f training_full.log"
echo "========================================"
