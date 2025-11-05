# 3DSAM-LLA
3D SAM with LLA(Loss-Less Adaptation) for Prompt-based Medical Image Segmentation

## 개요 (Abstract)

**3DSAM-LLA**는 TAVE 16기 MediVision 팀의 후반기 프로젝트로, 3D 의료 영상(CT, MRI 등)을 위한 SAM 성능 향상 연구를 주제로 합니다. 선행 연구인 3DSAM-adapter를 기반으로, 우리는 어댑터를 두 가지 경로로 나누는 Dual-path adapter 설계를 제안합니다. **Dual-path Adapter**는 기존 depth-wise 3D conv 대신, global branch(3D attention) + local branch(3D conv) 병렬 결합으로 다양한 공간 스케일 정보 보존하도록 설계한 어댑터 구조입니다. 이를 통해 세밀한 구조 정보 손실을 최소화하는 것을 기대합니다. 

## 진행 상태 (Current State)

현재 저희 프로젝트는 막 시작한 상태로, 이 레포지토리에는 baseline에 해당하는 3DSAM-adapter코드와, 저희 아이디어를 구현할 3DSAM-LLA 폴더 안에 기본적인 SAM 코드가 올라와 있습니다. 

## 폴더 설명 (Description)

- `3DSAM-LLA/` : 본 프로젝트의 핵심인 **Dual-path Adapter (LLA)**를 구현하는 메인 디렉토리입니다. 현재는 SAM의 기본 코드를 기반으로 하며, 향후 Global branch(3D attention)와 Local branch(3D conv)를 병렬 결합하는 저희의 고유한 어댑터 구조로 수정 및 개발이 진행될 공간입니다.

- `baseline/3DSAM-adapter/` : 성능 비교의 기준점이 되는 선행 연구, 3DSAM-adapter의 공식 코드가 위치한 디렉토리입니다. 저희가 제안하는 LLA 모델의 성능 향상 정도를 측정하기 위한 베이스라인으로 사용됩니다.
