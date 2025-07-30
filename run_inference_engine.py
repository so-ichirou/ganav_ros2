import zmq
import numpy as np
import cv2
from mmseg.apis import init_segmentor, inference_segmentor

def main():
    # --- モデルの読み込み ---
    # ご自身の環境に合わせてパスを修正してください
    config_file = '/home/labkubota/GANav-offroad/configs/ours/ganav_group6_rugd.py'
    checkpoint_file = '/home/labkubota/GANav-offroad/work_dirs/ganav_group6_rugd/latest.pth'
    device = 'cuda:0'
    
    print('Loading model...')
    model = init_segmentor(config_file, checkpoint_file, device=device)
    print('Model has been loaded.')

    # --- ZeroMQのセットアップ ---
    context = zmq.Context()
    socket = context.socket(zmq.REP) # REP (リプライ) 型のソケット
    socket.bind("tcp://*:5555")
    print("Inference engine is ready to receive images.")

    custom_palette_bgr = [
        [0, 0, 0],       # クラスID 0: 背景 -> 黒
        [0, 255, 0],     # クラスID 1: 滑らかな道 -> 緑
        [0, 255, 255],     # クラスID 2: 荒い道 -> 黄色
        [0, 165, 255],   # クラスID 3: でこぼこ -> オレンジ
        [0, 0, 255],     # クラスID 4: 進入不可 -> 赤
        [255, 0, 0]    # クラスID 5: 障害物 -> 青
    ]
    print("Using custom color palette.")

    # --- メインループ ---
    while True:
        # ROSラッパーから画像情報と画像データを受信
        frame_info = socket.recv_json()
        socket.send(b"OK")
        message = socket.recv()
        
        print("Received image, running inference...")

        actual_height, actual_width, _ = frame_info['shape']
        model_width, model_height = 688, 550

        cv_image_raw = np.frombuffer(message, dtype=np.uint8).reshape(actual_height, actual_width, 3)
        cv_image_resized = cv2.resize(cv_image_raw, (model_width, model_height))

        # --- 推論の実行 ---
        result = inference_segmentor(model, cv_image_resized)
        seg = result[0]

        # --- 結果の可視化 (カスタムパレットを使用) ---
        # palette = model.PALETTE  <-- この行はもう使いません
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        
        # 上で定義したカスタムパレットを使って色付けします
        for label, color in enumerate(custom_palette_bgr):
            color_seg[seg == label, :] = color
        
        # color_seg = cv2.cvtColor(color_seg, cv2.COLOR_RGB2BGR) <-- この行も不要になります
        
        # --- 結果をROSラッパーに返信 ---
        socket.send(color_seg.tobytes(), copy=False, track=False)
        print("Sent segmentation result.")

if __name__ == '__main__':
    main()
