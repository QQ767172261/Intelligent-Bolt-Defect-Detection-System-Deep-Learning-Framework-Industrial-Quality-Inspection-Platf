智慧工业**-基于YOLOv8的螺栓缺陷检测系统**![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4b3bf285897d4c619250c7725445da69.png)

---

# 🔩 螺栓缺陷智能检测系统 —— 基于 YOLOv8 的工业质检平台

## 一、项目概览表

| 项目维度       | 内容描述                                                                 |
|----------------|--------------------------------------------------------------------------|
| **项目名称**   | 基于深度学习的螺栓缺陷检测系统                                           |
| **核心算法**   | YOLOv8 目标检测                                                          |
| **缺陷类别**   | 10类：`鼓包`, `爆裂`, `圆形缺陷`, `裂纹`, `损伤`, `褶皱`, `划痕`, `工具痕迹`, `空洞`, `编码错误` |
| **数据集规模** | 451张标注图片（YOLO格式）                                                |
| **模型状态**   | 已训练好，开箱即用                                                       |
| **支持输入**   | 图片 / 视频 / 摄像头                                                     |
| **界面框架**   | PyQt5 图形化操作界面                                                     |
| **输出结果**   | JSON格式检测报告 + 可视化标注图                                          |
| **适用场景**   | 工业质检、生产线自动化、毕业设计、AI科研                                 |

---

## 二、功能特性矩阵表

| 功能编号 | 功能名称         | 详细描述                                                                 |
|----------|------------------|--------------------------------------------------------------------------|
| ①        | 单图检测         | 选择单张图片进行缺陷识别，实时显示框选结果与置信度                       |
| ②        | 批量检测         | 选择文件夹批量处理，表格化展示每张图的检测结果                           |
| ③        | 视频检测         | 支持本地视频文件逐帧分析，动态绘制缺陷位置                               |
| ④        | 摄像头实时检测   | 调用本地摄像头，毫秒级响应，适合产线部署                                 |
| ⑤        | 参数自定义       | 可调节置信度阈值、IOU阈值、选择模型、切换CPU/GPU                         |
| ⑥        | 标签与原图切换   | 可选是否显示标签名称、置信度、原始图像                                   |
| ⑦        | 结果保存         | 支持将检测结果保存为JSON文件，便于后续统计或导入Excel                    |
| ⑧        | 多模型支持       | 支持加载不同权重文件（如yolov8n.pt, best.pt等），灵活适配精度/速度需求    |

---

## 三、技术架构蓝图表

| 层级           | 技术组件                  | 作用说明                                                                 |
|----------------|---------------------------|--------------------------------------------------------------------------|
| **前端交互层** | PyQt5                     | 提供图形化操作界面，支持按钮点击、图像渲染、表格展示                     |
|                | OpenCV + QImage           | 图像预处理与Qt界面像素转换                                               |
|                | QThread                   | 多线程避免界面卡顿，保障视频流流畅                                     |
| **后端算法层** | Ultralytics YOLOv8        | 核心检测引擎，支持10类缺陷目标定位                                       |
|                | NMS后处理                 | 非极大值抑制，去除重复框                                                 |
|                | CPU/CUDA推理              | 支持硬件加速，适配不同算力设备                                           |
| **数据存储层** | 数据集 (451张)            | COCO/YOLO格式标注，涵盖10类缺陷                                          |
|                | 模型权重 (best.pt)        | 预训练模型，mAP@0.5 > 0.75（依训练而定）                                |
|                | 配置文件 (app_settings.json) | 保存用户偏好设置                                                         |
|                | 输出报告 (result.json)    | 结构化存储检测结果，含类别、坐标、置信度                                 |

---

## 四、最简易完整代码实现（单文件可运行版）

> 💡 以下代码整合了 UI + 检测核心 + 参数设置 + 结果保存，复制粘贴即可运行！

```python
# main.py - 螺栓缺陷检测系统（最简完整版）
import sys
import cv2
import json
import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from ultralytics import YOLO

class BoltDefectDetector:
    def __init__(self, model_path='models/best.pt', device='cpu'):
        self.model = YOLO(model_path).to(device)
        self.device = device
        self.names = self.model.names  # ['鼓包','爆裂',...]

    def detect_image(self, img_path, conf=0.25, iou=0.45):
        results = self.model.predict(img_path, conf=conf, iou=iou, device=self.device, verbose=False)
        result = results[0]
        img0 = cv2.imread(img_path)
        annotated = result.plot()
        
        detections = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy()
            for i in range(len(boxes)):
                x1,y1,x2,y2 = map(int, boxes[i])
                detections.append({
                    'class_id': int(cls_ids[i]),
                    'class_name': self.names[int(cls_ids[i])],
                    'confidence': float(confs[i]),
                    'bbox': [x1, y1, x2, y2]
                })
        return img0, annotated, detections

    def detect_frame(self, frame, conf=0.25, iou=0.45):
        results = self.model.predict(frame, conf=conf, iou=iou, device=self.device, verbose=False)
        result = results[0]
        annotated = result.plot()
        detections = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy()
            for i in range(len(boxes)):
                x1,y1,x2,y2 = map(int, boxes[i])
                detections.append({
                    'class_id': int(cls_ids[i]),
                    'class_name': self.names[int(cls_ids[i])],
                    'confidence': float(confs[i]),
                    'bbox': [x1, y1, x2, y2]
                })
        return frame, annotated, detections

class DetectionThread(QThread):
    signal = pyqtSignal(object, object, list, str)
    def __init__(self, detector, source, mode, conf, iou):
        super().__init__()
        self.detector = detector
        self.source = source
        self.mode = mode
        self.conf = conf
        self.iou = iou
        self.stop_flag = False

    def run(self):
        if self.mode == 'image':
            img0, ann, dets = self.detector.detect_image(self.source, self.conf, self.iou)
            self.signal.emit(img0, ann, dets, 'image')
        elif self.mode == 'video':
            cap = cv2.VideoCapture(self.source)
            while cap.isOpened() and not self.stop_flag:
                ret, frame = cap.read()
                if not ret: break
                img0, ann, dets = self.detector.detect_frame(frame, self.conf, self.iou)
                self.signal.emit(img0, ann, dets, 'video')
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            cap.release()
        elif self.mode == 'camera':
            cap = cv2.VideoCapture(0)
            while not self.stop_flag:
                ret, frame = cap.read()
                if not ret: break
                img0, ann, dets = self.detector.detect_frame(frame, self.conf, self.iou)
                self.signal.emit(img0, ann, dets, 'camera')
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            cap.release()

    def stop(self):
        self.stop_flag = True

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于深度学习的螺栓缺陷检测系统")
        self.setGeometry(100, 100, 1400, 900)

        self.detector = BoltDefectDetector()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()

        title = QLabel("基于深度学习的螺栓缺陷检测系统")
        title.setFont(QFont("微软雅黑", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        content = QHBoxLayout()

        # 左侧图像区
        left = QGroupBox("单位")
        left_layout = QVBoxLayout()
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("border:1px solid gray; min-height:400px;")
        self.img_label.setText("请点击‘打开图片’或‘打开视频’")
        left_layout.addWidget(self.img_label)
        left.setLayout(left_layout)

        # 右侧控制区
        right = QGroupBox("作者")
        right_layout = QVBoxLayout()

        # 参数设置
        param_group = QGroupBox("检测参数设置")
        param_form = QVBoxLayout()
        self.conf_input = QLineEdit("0.25")
        self.iou_input = QLineEdit("0.45")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["CPU", "CUDA"])
        param_form.addWidget(QLabel("置信度阈值:"))
        param_form.addWidget(self.conf_input)
        param_form.addWidget(QLabel("交并比阈值:"))
        param_form.addWidget(self.iou_input)
        param_form.addWidget(QLabel("检测设备:"))
        param_form.addWidget(self.device_combo)
        param_group.setLayout(param_form)

        # 结果显示
        result_group = QGroupBox("检测结果")
        result_form = QVBoxLayout()
        self.count_label = QLabel("目标数目: 0")
        self.type_label = QLabel("类型: --")
        self.conf_label = QLabel("置信度: --")
        self.pos_label = QLabel("位置: xmin=--, xmax=--, ymin=--, ymax=--")
        result_form.addWidget(self.count_label)
        result_form.addWidget(self.type_label)
        result_form.addWidget(self.conf_label)
        result_form.addWidget(self.pos_label)
        result_group.setLayout(result_form)

        # 结果表格
        table_group = QGroupBox("检测结果与位置信息")
        table_layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["序号", "文件路径", "类别", "置信度", "坐标"])
        table_layout.addWidget(self.table)
        table_group.setLayout(table_layout)

        # 操作按钮
        btn_layout = QHBoxLayout()
        self.btn_img = QPushButton("打开图片")
        self.btn_folder = QPushButton("打开文件夹")
        self.btn_video = QPushButton("打开视频")
        self.btn_camera = QPushButton("打开摄像头")
        self.btn_save = QPushButton("保存")
        self.btn_exit = QPushButton("退出")

        self.btn_img.clicked.connect(self.open_image)
        self.btn_folder.clicked.connect(self.open_folder)
        self.btn_video.clicked.connect(lambda: self.start_detection('video'))
        self.btn_camera.clicked.connect(lambda: self.start_detection('camera'))
        self.btn_save.clicked.connect(self.save_results)
        self.btn_exit.clicked.connect(self.close)

        btn_layout.addWidget(self.btn_img)
        btn_layout.addWidget(self.btn_folder)
        btn_layout.addWidget(self.btn_video)
        btn_layout.addWidget(self.btn_camera)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_exit)

        right_layout.addWidget(param_group)
        right_layout.addWidget(result_group)
        right_layout.addWidget(table_group)
        right_layout.addLayout(btn_layout)
        right.setLayout(right_layout)

        content.addWidget(left, stretch=1)
        content.addWidget(right, stretch=1)
        layout.addLayout(content)
        central.setLayout(layout)

        self.all_results = []
        self.thread = None

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.detect_single(path)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            self.all_results = []
            self.table.setRowCount(0)
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
            for idx, f in enumerate(files):
                full_path = os.path.join(folder, f)
                self.detect_single(full_path, idx+1)

    def detect_single(self, path, idx=None):
        img0, ann, dets = self.detector.detect_image(path, float(self.conf_input.text()), float(self.iou_input.text()))
        
        # 显示第一张图
        if idx is None or idx == 1:
            rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
            h,w,ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
            self.img_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.img_label.size(), Qt.KeepAspectRatio))
            
            if dets:
                d = dets[0]
                self.count_label.setText(f"目标数目: {len(dets)}")
                self.type_label.setText(f"类型: {d['class_name']}")
                self.conf_label.setText(f"置信度: {d['confidence']*100:.2f}%")
                x1,y1,x2,y2 = d['bbox']
                self.pos_label.setText(f"xmin={x1}, xmax={x2}, ymin={y1}, ymax={y2}")

        # 添加到表格
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(idx or row+1)))
        self.table.setItem(row, 1, QTableWidgetItem(os.path.basename(path)))
        self.table.setItem(row, 2, QTableWidgetItem(", ".join([d['class_name'] for d in dets])))
        self.table.setItem(row, 3, QTableWidgetItem(", ".join([f"{d['confidence']:.2f}" for d in dets])))
        self.table.setItem(row, 4, QTableWidgetItem(", ".join([str(d['bbox']) for d in dets])))

        self.all_results.append({'file': path, 'detections': dets})

    def start_detection(self, mode):
        if mode == 'video':
            source, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi)")
            if not source: return
        else:
            source = 0  # camera

        self.thread = DetectionThread(self.detector, source, mode, float(self.conf_input.text()), float(self.iou_input.text()))
        self.thread.signal.connect(self.on_result)
        self.thread.start()

    def on_result(self, img0, ann, dets, mode):
        rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.img_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.img_label.size(), Qt.KeepAspectRatio))

        self.count_label.setText(f"目标数目: {len(dets)}")
        if dets:
            d = dets[0]
            self.type_label.setText(f"类型: {d['class_name']}")
            self.conf_label.setText(f"置信度: {d['confidence']*100:.2f}%")
            x1,y1,x2,y2 = d['bbox']
            self.pos_label.setText(f"xmin={x1}, xmax={x2}, ymin={y1}, ymax={y2}")

    def save_results(self):
        if not self.all_results:
            QMessageBox.warning(self, "提示", "暂无检测结果！")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存结果", "", "JSON Files (*.json)")
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.all_results, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "成功", f"已保存至:\n{path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
```

---

## 五、运行说明表

| 步骤 | 操作内容                     | 命令/说明                                                                 |
|------|------------------------------|---------------------------------------------------------------------------|
| 1    | 安装依赖                     | `pip install ultralytics opencv-python PyQt5 torch torchvision`          |
| 2    | 准备模型                     | 下载 `best.pt` 放入 `models/` 目录                                        |
| 3    | 运行程序                     | `python main.py`                                                          |
| 4    | 使用界面                     | 点击“打开图片”→选择图片→查看结果→点击“保存”导出JSON                      |
| 5    | 批量检测                     | 点击“打开文件夹”→选择含多图文件夹→自动填充表格                            |
| 6    | 实时检测                     | 点击“打开摄像头”→实时画面中绘制缺陷框                                    |

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5a58e1ef26854eb1b256ae38f1d42348.png)

## 六、交付清单表

| 文件类型             | 内容说明                                     |
|----------------------|----------------------------------------------|
| 全部源码 (.py)       | 包含主程序、UI逻辑、检测核心                 |
| UI源文件 (.ui/.qrc)  | 界面设计文件，可用Qt Designer修改            |
| 数据集 (451张)       | 标注好的YOLO格式图片，10类缺陷               |
| 训练模型 (best.pt)   | 预训练权重，可直接用于推理                   |
| 训练结果             | PR曲线、混淆矩阵、损失图等                   |
| 运行教程             | 图文/PDF指南，含环境配置、常见问题解答       |
| 环境配置             | requirements.txt / installPackages.bat       |
| 示例数据             | 测试图片/视频，用于快速验证                  |

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7fc8e4ca1e924ca6aa9350315f9697c3.png)

## 七、扩展建议表

| 扩展方向       | 实现方式                                     | 应用场景                     |
|----------------|----------------------------------------------|------------------------------|
| 缺陷分类增强   | 增加更多缺陷子类（如锈蚀、变形）             | 更精细质检                   |
| 尺寸测量       | 结合标定板计算缺陷实际尺寸                   | 工艺合规性判断               |
| 报警联动       | 检测到严重缺陷时触发声光报警                 | 产线自动控制                 |
| 数据库对接     | 将结果存入MySQL/SQLite                       | 质量追溯系统                 |
| Web端部署      | Flask + Vue 构建浏览器访问界面               | 远程监控                     |
| 多相机同步     | 支持多路摄像头并发检测                       | 多角度覆盖                   |
| 模型热更新     | 界面内一键切换不同版本模型                   | A/B测试、迭代优化            |

---

## 八、总结评价表

| 维度         | 评分（⭐️⭐️⭐️⭐️⭐️） | 说明                                                                 |
|--------------|---------------------|----------------------------------------------------------------------|
| 学术价值     | ⭐️⭐️⭐️⭐️☆           | 提供完整训练流程、评估指标、可视化分析                               |
| 工程价值     | ⭐️⭐️⭐️⭐️⭐️           | 模块化设计、跨平台、易部署、可扩展                                   |
| 应用价值     | ⭐️⭐️⭐️⭐️⭐️           | 直接服务于工业质检、智能制造、毕业设计、AI科研                       |
| 易用性       | ⭐️⭐️⭐️⭐️⭐️           | 图形界面友好，零基础也可上手                                         |
| 性能表现     | ⭐️⭐️⭐️⭐️☆           | CPU/GPU均支持，451张数据集训练收敛快，推理速度快                     |

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/65d7bb7695754687997fc2ae1c121511.png)

