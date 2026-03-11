# Advanced Face and Hand Tracking System

A professional biometric analysis system featuring **gesture-controlled face mesh visualization** and **real-time hand tracking** with a futuristic sci-fi interface.

![System Demo](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-red)

## 🎯 Features

### **Gesture-Controlled Face Mesh**
- **Dense green point cloud** face visualization (Matrix-style)
- **Hand gesture controls** for real-time mesh manipulation
- **Professional biometric interface** with 3D positioning
- **Dynamic mesh density** based on finger gestures

### **Advanced Hand Tracking**
- **Real-time gesture recognition** (6 different gestures)
- **Finger position analysis** with debug information
- **Confidence-based control** system
- **Visual feedback** for gesture status

### **Professional Interface**
- **Dark grid background** for technical appearance
- **Clean, minimal design** without text clutter
- **Fullscreen mode** for immersive experience
- **Real-time status display** with gesture guide

## 🖐️ Gesture Controls

| Gesture | Fingers | Effect | Description |
|---------|---------|--------|-------------|
| ✊ **FIST** | All closed | Minimal Mesh | 20% density, small points |
| 👆 **POINT** | Index only | Low Density | 40% density, medium points |
| ✌️ **PEACE** | Index + Middle | Medium Density | 60% density, balanced view |
| 🤟 **THREE** | Thumb + Index + Middle | High Density | 80% density, detailed view |
| 🖖 **FOUR** | 4 fingers (no thumb) | Maximum Density | 100% density, large points |
| 🖐️ **OPEN** | All 5 fingers | Full Features | 100% + contours + key points |

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam or camera device
- Good lighting conditions

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
```
opencv-python==4.8.1.78
mediapipe==0.10.14
numpy==1.24.3
```

## 💻 Usage

### Basic Usage
```bash
python posture_advanced.py
```

### System Startup
1. **Camera initialization** - System detects your webcam
2. **MediaPipe loading** - Face and hand detection models load
3. **Fullscreen mode** - Professional interface launches
4. **Gesture recognition** - Start using hand gestures to control face mesh

### Controls
| Key | Function |
|-----|----------|
| `Q` or `ESC` | Exit application |
| `SPACE` | Toggle fullscreen mode |
| `S` | Save screenshot |
| `R` | Reset statistics |
| `H` | Toggle hand detection |
| `F` | Toggle face detection |

## 🔧 Configuration

### Camera Settings
```python
CONFIG = {
    "camera_id": 0,           # 0 for default webcam
    "frame_width": 640,       # Camera resolution width
    "frame_height": 480,      # Camera resolution height
    "mirror_mode": True,      # Flip frame horizontally
}
```

### Detection Settings
```python
CONFIG = {
    "detection_confidence": 0.5,    # Face/hand detection confidence
    "tracking_confidence": 0.5,     # Tracking confidence
    "enable_hand_detection": True,  # Enable gesture control
    "enable_face_detection": True,  # Enable face mesh
}
```

### Face Mesh Settings
```python
face_mesh_config = {
    "show_face_mesh": True,     # Show/hide face mesh
    "mesh_density": 1.0,        # Point density (0.0-1.0)
    "point_size": 2,            # Point size (1-4)
    "show_contours": False,     # Show mesh lines
    "show_key_points": False,   # Show key landmarks
}
```

## 🎨 Visual Components

### Face Mesh Visualization
- **Position**: Left side of screen
- **Color**: Matrix-style green (#00FF00)
- **Scale**: Adjustable (default 2.2x)
- **Density**: Gesture-controlled (20%-100%)

### Gesture Status Display
- **Position**: Top-right corner
- **Information**: Current gesture, mesh status, finger debug
- **Guide**: Real-time gesture control reference

### Background
- **Style**: Professional dark grid
- **Opacity**: 80% dark overlay
- **Grid**: Subtle technical lines

## 🔍 Technical Details

### Gesture Recognition Algorithm
1. **Landmark extraction** from MediaPipe hand model
2. **Finger position analysis** using tip/joint relationships
3. **Gesture classification** based on extended finger count
4. **Confidence scoring** for reliable detection
5. **Real-time application** to face mesh parameters

### Face Mesh Rendering
1. **468 facial landmarks** from MediaPipe Face Mesh
2. **3D depth calculation** for realistic positioning
3. **Dynamic point sizing** based on depth and gesture
4. **Interpolated density** for smooth transitions
5. **Optimized rendering** for real-time performance

### Performance Optimizations
- **Selective landmark processing** based on density
- **Efficient gesture detection** with confidence thresholds
- **Optimized rendering loops** for smooth frame rates
- **Memory management** for continuous operation

## 📊 System Requirements

### Minimum Requirements
- **CPU**: Intel i3 / AMD Ryzen 3 or equivalent
- **RAM**: 4GB
- **Camera**: 720p webcam
- **OS**: Windows 10, macOS 10.14, Ubuntu 18.04

### Recommended Requirements
- **CPU**: Intel i5 / AMD Ryzen 5 or better
- **RAM**: 8GB or more
- **Camera**: 1080p webcam with good low-light performance
- **OS**: Latest versions for optimal MediaPipe performance

## 🐛 Troubleshooting

### Common Issues

#### Face Mesh Not Appearing
- Check if face detection is enabled (`F` key)
- Ensure good lighting conditions
- Verify camera is working properly
- Try different gesture (open hand for full features)

#### Gesture Recognition Problems
- Check finger debug display for detection status
- Ensure hand is clearly visible to camera
- Try exaggerated finger positions
- Verify lighting is adequate

#### Performance Issues
- Close other camera applications
- Reduce mesh density with fist gesture
- Check system resources
- Update graphics drivers

#### Camera Not Found
```bash
# Check available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).read()[0]])"
```

### Debug Information
The system displays real-time debug information including:
- Current gesture recognition
- Individual finger status (UP/DOWN)
- Mesh density and point size
- Detection confidence levels

## 🔬 Applications

### Professional Use Cases
- **Biometric Analysis**: Advanced facial feature mapping
- **Security Systems**: Gesture-based access control
- **Medical Applications**: Hand mobility assessment
- **Research**: Human-computer interaction studies
- **Education**: Anatomy and gesture recognition learning

### Development Extensions
- **Custom Gestures**: Add new gesture patterns
- **Data Export**: Save mesh data for analysis
- **Multi-user**: Support multiple faces/hands
- **Integration**: Connect with other systems via API

## 📝 File Structure

```
├── posture_advanced.py      # Main application
├── face_mesh.py            # Basic face mesh functions
├── face_mesh_enhanced.py   # Enhanced face mesh features
├── posture.py             # Original posture detection
├── requirements.txt       # Python dependencies
├── posture_log.json      # Session logging data
├── POSTURE_README.md     # Detailed documentation
└── README.md            # This file
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings for functions
- Include type hints where appropriate
- Test with different camera setups

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe Team** - For excellent hand and face detection models
- **OpenCV Community** - For computer vision tools
- **Python Community** - For the amazing ecosystem

## 📞 Support

For issues, questions, or contributions:
- Create an issue in the repository
- Check troubleshooting section above
- Review MediaPipe documentation for model details

---

**Made with ❤️ for advanced biometric analysis and human-computer interaction**