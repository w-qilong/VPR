import cv2
import numpy as np

def select_position(image, target_size=None):
    """
    显示图像并获取鼠标点击位置的坐标列表
    
    Args:
        image: 输入图像
        target_size: 目标尺寸，格式为(width, height)。如果为None则使用原始尺寸
    
    Returns:
        clicked_points: 所有点击位置的坐标列表[(x1, y1), (x2, y2), ...]，如果用户未点击任何位置就按ESC则返回空列表
    """
    cv2.destroyAllWindows()
    
    if target_size is not None:
        display_image = cv2.resize(image, target_size)
    else:
        display_image = image.copy()
    
    clicked_points = []  # 改用列表存储所有点击位置
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append((x, y))
            print(f"点击位置坐标: ({x}, {y})")
            # 可以选择在图像上标记点击位置
            cv2.circle(display_image, (x, y), 3, (0, 0, 255), -1)
    
    window_name = "Select Position"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        cv2.imshow(window_name, display_image)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # 只有按ESC键才退出
            break
    
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return clicked_points

# 使用示例
if __name__ == "__main__":
    # 读取图像
    img = cv2.imread("sample_imgs/msls/3/query/oYwU28V-MzaKyX9R18zlMQ.jpg")
    
    # 设置目标尺寸
    target_size = (560, 560)
    
    # 获取所有点击位置
    points = select_position(img, target_size)
    
    if points:
        print(f"所有点击位置: {points}")
        print(f"总共点击了 {len(points)} 个位置")
    else:
        print("未选择任何位置")
