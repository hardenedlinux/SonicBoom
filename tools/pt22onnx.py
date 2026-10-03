import os
import argparse
import torch

def convert_pt2_to_onnx(pt2_path):
    # 1. 
    if not os.path.exists(pt2_path):
        print(f"  -> {pt2_path}")
        return

    # 2.  ONNX 
    base_name, _ = os.path.splitext(pt2_path)
    onnx_path = f"{base_name}.onnx"

    print(f"  PT2 : {pt2_path} ...")
    
    # 3.  ExportedProgram 
    exported_program = torch.export.load(pt2_path)
    
    # 4. 
    example_inputs = exported_program.example_inputs
    print(f"  {len(example_inputs)} ")
    for i, inp in enumerate(example_inputs):
        if isinstance(inp, torch.Tensor):
            print(f"   ->  [{i}]:  {list(inp.shape)},  {inp.dtype}")

    print(f" [SonicBoom]  ExportedProgram  ONNX : {onnx_path} ...")
    
    # 5.  model exported_program
    #  ONNX  TreeSpec  Tracing 
    onnx_program = torch.onnx.export(
        exported_program,
        input_names=["input"],
        output_names=["output"]
    )
    
    # 6.  ONNX 
    onnx_program.save(onnx_path)
    print(f"  ONNX : {onnx_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SonicBoom  .pt2  .onnx")
    parser.add_argument("pt2_file", type=str, help=" .pt2 resnet18.pt2")
    args = parser.parse_args()

    convert_pt2_to_onnx(args.pt2_file)
