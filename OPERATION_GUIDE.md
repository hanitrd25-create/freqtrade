# Freqtrade 운영 가이드

이 문서는 Freqtrade 설치 후, 봇을 효과적으로 운영하고 관리하기 위한 주요 절차와 팁을 안내합니다.

---

## 1. 설정 파일 (`config.json`) 주요 항목 관리

`user_data/config.json` 파일은 Freqtrade 봇의 모든 행동을 정의하는 가장 중요한 파일입니다. 주요 항목은 다음과 같습니다.

### 필수 설정

- `"max_open_trades"`: 동시에 진행할 최대 거래 수를 지정합니다. `3`으로 설정하면 최대 3개의 코인만 동시에 보유합니다.
- `"stake_currency"`: 거래의 기준이 되는 통화입니다. (예: `"USDT"`, `"BTC"`)
- `"stake_amount"`: 한 번의 거래에 사용할 금액입니다. `"unlimited"`로 설정하면 모든 잔액을 사용하며, 특정 금액(예: `100`)을 지정할 수도 있습니다.
- `"dry_run"`: `true`로 설정하면 실제 돈을 사용하지 않는 모의 투자를 진행합니다. **실제 돈을 투입하기 전에 반드시 `true`로 설정하여 충분히 테스트하세요.** `false`로 변경하면 실제 거래가 시작됩니다.

### 거래소 API 설정

`"exchange"` 섹션에서 거래소 API 키를 설정해야 실제 계정과 연동됩니다.

```json
"exchange": {
    "name": "binance",
    "key": "YOUR_API_KEY",
    "secret": "YOUR_API_SECRET",
    "ccxt_config": {},
    "ccxt_async_config": {}
},
```
- `"key"`: 거래소에서 발급받은 API Key를 입력합니다.
- `"secret"`: 거래소에서 발급받은 API Secret을 입력합니다.

### 텔레그램 봇 연동

텔레그램으로 봇의 상태를 보고받고 제어할 수 있습니다.

```json
"telegram": {
    "enabled": true,
    "token": "YOUR_TELEGRAM_TOKEN",
    "chat_id": "YOUR_TELEGRAM_CHAT_ID"
},
```
- `"enabled"`: `true`로 설정하여 활성화합니다.
- `"token"`: BotFather로부터 발급받은 텔레그램 봇 토큰을 입력합니다.
- `"chat_id"`: 봇과 대화를 시작한 후, 당신의 고유 chat ID를 확인하여 입력합니다.

### FreqUI 보안 설정

`"api_server"` 섹션에서 UI 접속을 위한 사용자 이름과 비밀번호를 설정할 수 있습니다.

```json
"api_server": {
    "enabled": true,
    "listen_ip_address": "0.0.0.0",
    "listen_port": 8080,
    "username": "freqtrader",
    "password": "YOUR_STRONG_PASSWORD"
},
```
- `"username"` / `"password"`: FreqUI 로그인 시 사용할 아이디와 비밀번호를 설정합니다.

---

## 2. 전략 (`Strategy`) 관리

- **전략 파일 위치**: 모든 전략 파일은 `user_data/strategies/` 디렉토리에 위치해야 합니다.
- **전략 변경**: 봇에 적용할 전략을 변경하려면 `docker-compose.yml` 파일의 `command` 섹션에서 `--strategy` 부분을 수정해야 합니다.

  ```yaml
  # docker-compose.yml
  command: >
    trade
    # ...
    --strategy YourNewStrategy # <--- 이 부분을 변경
  ```
  파일 수정 후에는 `docker compose up -d` 명령어로 봇을 다시 시작해야 적용됩니다.

- **커뮤니티 전략 활용**: Freqtrade 커뮤니티에서는 다양한 무료/유료 전략을 공유합니다. GitHub 등에서 `.py` 전략 파일을 다운로드하여 `user_data/strategies/`에 추가하고 테스트해볼 수 있습니다.

---

## 3. 데이터 다운로드 및 백테스팅

전략이 과거 데이터에서 어떤 성과를 냈는지 확인하는 백테스팅은 매우 중요합니다.

### 데이터 다운로드

Binance 거래소에서 2023년 1월 1일부터 현재까지의 BTC/USDT 5분봉 데이터를 다운로드하는 예시입니다.

```bash
docker compose run --rm freqtrade download-data \
--exchange binance \
--pairs BTC/USDT \
--days 500 \
-t 5m
```

### 백테스팅 실행

다운로드한 데이터를 사용하여 `SampleStrategy` 전략을 테스트하는 명령어입니다.

```bash
docker compose run --rm freqtrade backtesting \
--config user_data/config.json \
--strategy SampleStrategy
```

백테스팅 결과는 터미널에 표 형태로 요약되어 나타나며, 이를 통해 전략의 수익률, 승률 등을 평가할 수 있습니다.

---

## 4. 봇 실행 및 모니터링

### Docker를 이용한 봇 관리

- **봇 시작 (백그라운드 실행)**:
  ```bash
  docker compose up -d
  ```

- **봇 중지**:
  ```bash
  docker compose down
  ```

- **실시간 로그 확인**:
  ```bash
  docker compose logs -f
  ```

- **실행 중인 서비스 확인**:
  ```bash
  docker compose ps
  ```

### FreqUI를 통한 모니터링

웹 브라우저에서 `http://localhost:8080`으로 접속하면 대시보드를 통해 현재 진행 중인 거래, 자산 현황, 로그, 일별 수익률 등 상세한 정보를 시각적으로 확인할 수 있습니다.

---

## 5. 자주 사용하는 명령어 (텔레그램)

텔레그램 봇을 연동했다면, 아래와 같은 명령어로 봇을 제어할 수 있습니다.

- `/status table`: 현재 진행 중인 모든 거래를 표 형태로 보여줍니다.
- `/profit`: 모든 완료된 거래의 누적 수익률을 보여줍니다.
- `/daily`: 지난 7일간의 일별 수익률을 보여줍니다.
- `/balance`: 거래소에 보유 중인 자산 현황을 보여줍니다.
- `/forceexit [trade_id]`: 특정 거래를 즉시 강제로 종료(매도)합니다. `[trade_id]`는 `/status` 명령어로 확인할 수 있습니다.
- `/stop`: 봇을 정지합니다. (거래 중인 코인은 매도하지 않습니다.)
- `/start`: 정지된 봇을 다시 시작합니다.

이 가이드가 Freqtrade를 성공적으로 운영하는 데 도움이 되기를 바랍니다.
