第3版6类标志牌角点检测

当前精度最高的版本，速度1080Ti: 8-9FPS


环境配置，运行

1.pytorch1.3+

2. 编译detectron2
   cd demo/detectron2
   python setup.py build develop

3. 编译HRNet
   cd lib/
   make 

4. run
   python demo/inferenceSigncorner.py
