# 멀티코어 컴퓨팅 (2020년도, 1학기, 4190.414A) 기말 프로젝트

## A

cpu-only 브랜치를 참고하세요.

openmp 사용. 스레드 16개. 하나당 이미지 하나씩 처리.  
각종 루프 수정.  
conv2d, conv2d_transposed 등을 포함하여 곳곳에 벡터 익스텐션.  

32장 기준 4.5

## B

master 브랜치를 참고하세요. B C D 코드는 전부 같고 Makefile 만 수정했습니다.  
처음부터 C를 염두에 두고 짰습니다.

입력 이미지를 GPU에 올리고 모든 작업을 GPU에서 함으로써 메모리 이동 최소화.  
필터 등 계속 쓰이는 것은 한 번 GPU 버퍼에 올리고 재사용.  
커널 최적화 x 정직하게 A의 conv2d 옮김(ㅜㅜ)

128장 기준 4

## C

openmp로 스레드 4개를 만들고 위의 것을 각 스레드가 각자의 GPU에 대해서 하도록 만듦.

128장 기준 16

## D

처음에 모든 이미지를 골고루 4로 나눈 다음 (10 -> 3 3 2 2) 각 노드는 C가 한 것처럼 4개로 나눠서 돌림

지금 생각해보면 rank0이 더 적은 이미지를 가져야 할 것 같이 보임. 1번이 빨리 받아서 빨리 하나 처리하기?
