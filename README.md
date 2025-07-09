# YOLOX-TensorRT10

This project provides a C++ TensorRT implementation of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) with dynamic shape support. Although the YOLOX repository already provides a C++ TensorRT 8 [example](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/TensorRT/cpp/yolox.cpp), it only supports fixed-size inputs and has issues like memory leaks. That is why this project was created.



## Demo

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Avafly/ImageHostingService@master/uPic/YOLOX-TensorRT10-Demo.jpg" width = "500">
</p>



## Export a Dynamic‐Shape ONNX Model

To enable dynamic axes, modify `tools/export_onnx.py`:

```python
torch.onnx.export(
    model,
    dummy_input,
    args.output_name,
    input_names=[args.input],
    output_names=[args.output],
    # dynamic_axes={args.input: {0: 'batch'},
    #               args.output: {0: 'batch'}} if args.dynamic else None,
    dynamic_axes={args.input: {0: 'batch', 2: 'height', 3: 'width'},
                  args.output: {0: 'batch', 1: 'anchors'}} if args.dynamic else None,
    opset_version=args.opset,
)
```

Then run:

```bash
python3 tools/export_onnx.py --dynamic -n yolox_x -c models/yolox_x.pth
```

Convert onnx to engine model as follows.

```bash
trtexec --onnx=yolox_x.onnx --saveEngine=yolox_x.engine --fp16 --minShapes=images:1x3x32x32 --optShapes=images:1x3x640x640 --maxShapes=images:1x3x1280x1280
```



## Dependencies

**TensorRT:** 10.11

**CUDA:** 12.9

**OpenCV:** 4.12.0

