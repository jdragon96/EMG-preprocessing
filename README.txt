0. 13명의 sEMG데이터 cylinder, rock, pencil, hook 동작 데이터

1. loading
csv파일을 읽어오는 모듈

2. preprocessing
sEMG데이터 전처리 모듈
moving average와 FFT, STFT 사용 가능
* 추가로 필요한건 알아서 추가

3. model
CNN의 가장 기본적인 모델을 활용
데이터 사이즈가 50X50을 기준으로 오버피팅이 발생되지 않는 조건을 기본 조건으로 만들어놈

데이터 모델 변경 시 train loss, validation loss를 비교해 overfitting 방지할것