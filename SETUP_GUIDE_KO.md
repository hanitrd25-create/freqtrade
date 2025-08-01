# Freqtrade M4 MacBook Pro 설치 가이드

안녕하세요! 이 가이드는 Apple M4 칩이 탑재된 MacBook Pro에서 Freqtrade를 설치하고 실행하는 과정을 안내합니다. M4와 같은 ARM64 아키텍처 시스템에서는 공식적으로 Docker를 사용한 설치를 권장하고 있으므로, 이 가이드도 Docker를 기반으로 작성되었습니다.

---

## 사전 준비 사항

가장 먼저, Mac에 Docker Desktop이 설치되어 있어야 합니다. 아래 링크에서 다운로드하여 설치해 주세요.

- **[Docker Desktop for Mac 다운로드](https://docs.docker.com/desktop/install/mac-install/)**

Docker가 성공적으로 설치되고 실행 중인지 확인한 후에 다음 단계를 진행해 주세요.

---

## 설치 절차

### 1단계: 작업 디렉토리 생성 및 이동

Freqtrade 관련 파일들을 보관할 디렉토리를 만들고, 터미널에서 해당 디렉토리로 이동합니다.

```bash
mkdir freqtrade-m4
cd freqtrade-m4
```

### 2단계: `docker-compose.yml` 파일 다운로드

Freqtrade를 Docker 환경에서 실행하기 위한 설정 파일입니다. 아래 명령어를 터미널에 입력하여 다운로드합니다.

```bash
curl https://raw.githubusercontent.com/freqtrade/freqtrade/stable/docker-compose.yml -o docker-compose.yml
```

### 3단계: Freqtrade Docker 이미지 다운로드

Freqtrade의 최신 안정 버전을 Docker 이미지로 가져옵니다.

```bash
docker compose pull
```

### 4단계: 사용자 데이터 디렉토리 생성

전략, 설정 파일, 로그 등을 저장할 `user_data` 디렉토리를 생성합니다.

```bash
docker compose run --rm freqtrade create-userdir --userdir user_data
```

이 명령어를 실행하면 현재 디렉토리(`freqtrade-m4`) 내에 `user_data` 폴더가 생성된 것을 확인할 수 있습니다.

### 5단계: 설정 파일(`config.json`) 생성

봇 운영에 필요한 기본 설정 파일을 생성합니다. 아래 명령어를 실행하면 몇 가지 질문이 나타납니다. 답변을 통해 초기 설정을 구성할 수 있습니다.

```bash
docker compose run --rm freqtrade new-config --config user_data/config.json
```

**주요 질문:**
- `Do you want to enable Dry-run?` (모의 투자 활성화 여부): 처음에는 `y` (Yes)를 선택하여 실제 자금 없이 테스트하는 것을 강력히 권장합니다.
- `What is your stake currency?` (기본 통화): `USDT` 또는 `BTC` 등을 입력합니다.
- `What is your stake amount?` (거래당 투자 금액): `unlimited` 또는 원하는 금액을 입력합니다.
- `What is the name of your strategy?`: `SampleStrategy` (기본값)를 그대로 사용하거나 원하는 전략 이름을 입력합니다.
- `Enable FreqUI?`: `y`를 선택하면 웹 인터페이스를 사용할 수 있습니다.

생성된 설정 파일은 `user_data/config.json` 경로에서 언제든지 직접 수정할 수 있습니다.

### 6단계: (선택 사항) 사용자 정의 전략 추가

자신만의 거래 전략을 사용하고 싶다면, 파이썬으로 작성된 전략 파일(`your_strategy.py`)을 `user_data/strategies/` 디렉토리 안에 복사해 넣으면 됩니다.

그 후, `docker-compose.yml` 파일을 열어 `command` 부분을 다음과 같이 수정해야 합니다.

```yaml
# docker-compose.yml 파일 일부

services:
  freqtrade:
    image: freqtradeorg/freqtrade:stable
    # ... (다른 설정들) ...
    command: >
      trade
      --logfile /freqtrade/user_data/logs/freqtrade.log
      --config /freqtrade/user_data/config.json
      --strategy YourAwesomeStrategy # <--- 여기를 당신의 전략 클래스 이름으로 변경
```

### 7단계: Freqtrade 봇 시작하기

이제 모든 준비가 끝났습니다. 아래 명령어로 Freqtrade 봇을 백그라운드에서 실행합니다.

```bash
docker compose up -d
```

### 8단계: 봇 모니터링 및 관리

- **실행 상태 확인**:
  ```bash
  docker compose ps
  ```
  `freqtrade` 서비스가 `running` 상태로 표시되어야 합니다.

- **실시간 로그 확인**:
  ```bash
  docker compose logs -f
  ```
  (로그 확인을 중단하려면 `Ctrl + C`를 누르세요)

- **웹 UI 접속**:
  설정 과정에서 FreqUI를 활성화했다면, 웹 브라우저에서 `http://localhost:8080` 주소로 접속하여 봇의 상태를 확인할 수 있습니다.

- **봇 중지**:
  ```bash
  docker compose down
  ```

---

## Freqtrade 업데이트

Freqtrade를 최신 버전으로 업데이트하려면 아래 두 명령어를 차례로 실행하면 됩니다.

```bash
# 1. 최신 이미지를 다운로드합니다.
docker compose pull

# 2. 최신 이미지로 봇을 다시 시작합니다.
docker compose up -d
```

---

이 가이드가 Freqtrade를 성공적으로 설치하고 운영하는 데 도움이 되기를 바랍니다. 궁금한 점이 있다면 언제든지 질문해 주세요.
