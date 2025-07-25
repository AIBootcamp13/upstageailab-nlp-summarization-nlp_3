import torch

def main():
    print("Hello from gx-train!")
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    print(f"CUDA 버전: {torch.version.cuda}")

    if torch.cuda.is_available():
        print(f"GPU 개수: {torch.cuda.device_count()}")
        print(f"현재 GPU: {torch.cuda.get_device_name(0)}")
        
        # 간단한 GPU 연산 테스트
        x = torch.randn(3, 3).cuda()
        print(f"GPU 텐서 생성 성공: {x.device}")

if __name__ == "__main__":
    main()
