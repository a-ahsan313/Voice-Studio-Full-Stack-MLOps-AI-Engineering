import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="RVC Inference CLI")
    parser.add_argument("--model", type=str, required=True, help="Path to .pth model file")
    parser.add_argument("--index", type=str, required=False, default="", help="Path to .index file")
    parser.add_argument("--input", type=str, required=True, help="Path to input audio")
    parser.add_argument("--output", type=str, required=True, help="Path to output audio")
    parser.add_argument("--pitch", type=int, default=0, help="Pitch adjustment (semitones)")
    parser.add_argument("--method", type=str, default="rmvpe", help="f0 method (rmvpe, pm, harvest)")
    args = parser.parse_args()

    # Import inside main to save time if help/error
    from rvc_python.infer import VoiceClone
    
    try:
        # Initialize the voice clone pipeline
        vc = VoiceClone()
        
        # Load the model
        vc.load_model(args.model)
        
        # Perform inference
        vc.infer(
            input_path=args.input,
            output_path=args.output,
            f0_up_key=args.pitch,
            f0_method=args.method,
            index_path=args.index if args.index else None
        )
        print(f"✅ RVC generation successful: {args.output}")
        sys.exit(0)
    except Exception as e:
        print(f"❌ RVC Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
