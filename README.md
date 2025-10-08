# Snake-ML Python-port

Det här projektet är en Python-port av Marcus Peterssons Snake-ML med stöd för Gymnasium, Stable-Baselines3, PyTorch och ONNX-export. Repositoriet återskapar hela logiken från webbläsarversionen och gör det möjligt att träna ormen i realtid via `pygame`, köra flera miljöer parallellt och exportera modeller till Snake-ML:s "Watch"-läge.

## Förutsättningar

* Python 3.10+ (3.11 rekommenderas)
* `pip` och `virtualenv`/`venv`
* Systempaket för att kunna kompilera PyTorch och använda `pygame` (exempelvis `sudo apt-get install python3-dev python3-venv build-essential libSDL2-dev` på Debian/Ubuntu)

## Snabbstart

```bash
cd snakepython
./install.sh        # skapar virtuell miljö .venv/ och installerar dependencies
source .venv/bin/activate
python train_dqn.py # startar DQN-träning med realtidsrendering
```

> Tips: Lägg till flaggan `--tensorboard` till träningsskripten för att aktivera TensorBoard-loggning under `./tb_snake/`.

## Spela direkt i IDLE (utan extra installationer)

Vill du bara spela ett klassiskt Snake utan att installera något? Öppna filen
`snakepython/idle_snake.py` i IDLE och kör `Run > Run Module`. Spelet använder
en enkel `tkinter`-ruta, styrs med piltangenterna och håller reda på poäng,
insamlade frukter och antal steg. Tryck `Space` för att starta om och `Escape`
för att stänga fönstret.

### Träna Double DQN direkt från IDLE

`idle_snake.py` innehåller nu en hel Double DQN-implementation skriven i ren
Python/NumPy. Du kan träna, spara och spela upp agenten utan att lämna IDLE:

```bash
python idle_snake.py --train 1000 --steps 600 --save-model idle_dqn.npz
```

* `--train` anger hur många episoder som ska köras.
* `--steps` sätter maxsteg per episod.
* `--save-model` sparar vikterna i ett `.npz`-paket.
* `--load-model` kan användas för att återuppta träning eller spela med en
  tidigare tränad agent.

Vill du titta på agenten efter träning kan du starta spelet med autopilotläget
påslaget:

```bash
python idle_snake.py --load-model idle_dqn.npz --play --autopilot
```

Terminalen skriver ut träningsstatistik var tionde episod och autopilot kan
slås av/på i spelet genom att trycka på `A`.

## Manuella installationssteg

Föredrar du att göra allt manuellt kan du följa dessa steg:

```bash
cd snakepython
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Träna modeller

| Script | Beskrivning | Standardparametrar |
| ------ | ----------- | ------------------ |
| `train_dqn.py` | Tränar en DQN-agent med valfritt antal parallella miljöer (`--parallel-envs`). Renderar miljö 0 i realtid. | 500 000 steg, `CnnPolicy`, `tensorboard_log="./tb_snake/"` när flaggan används. |
| `train_ppo.py` | Tränar en PPO-agent med samma miljö och motsvarande loggning. | `learning_rate=3e-4`, `gamma=0.975`, `n_steps=2048`, m.fl. |

Samtliga skript tar emot följande vanliga flaggor:

* `--timesteps <int>` – antal träningssteg (standard 500_000).
* `--grid-size <int>` – rutnätsstorlek (10–20 rekommenderas).
* `--tensorboard` – aktivera TensorBoard-loggar.
* `--seed <int>` – sätt slumpfrö.
* `--parallel-envs <int>` – antal miljöer som körs samtidigt (standard 8).

### Multi-run launcher

För att starta flera oberoende träningssessioner (t.ex. på olika seeds) kan du använda `utils/run_multi_train.py`:

```bash
python utils/run_multi_train.py --runs 4 --algo dqn --timesteps 200000
```

Detta skapar fyra processer som var och en sparar modeller i `models/<algo>_snake_runX.zip` och loggar till `tb_snake/runX/`.

## Utvärdera en modell

När träningen är klar kan du spela upp agenten i tio episoder:

```bash
python evaluate.py --model models/dqn_snake_<timestamp>.zip
```

Fönstret visar ormen live och terminalen skriver `Episode N | Reward: X | Fruits: Y | Steps: Z`.

## Export till ONNX och JSON

```bash
python export_model.py --model models/dqn_snake_<timestamp>.zip
```

Skriptet skapar:

* `export/snake_agent.onnx` – ONNX-modellen (inputformat `[1, 3, grid_size, grid_size]`).
* `export/snake_agent.json` – meta- och viktinformation som kan laddas i Snake-ML:s webbläsargränssnitt.

## Integration med Snake-ML-webben

Lägg till följande i webbkoden:

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
<button id="loadPythonModelBtn">🧠 Load Trained Model (Python)</button>
```

När knappen klickas:

```javascript
const session = await ort.InferenceSession.create('export/snake_agent.onnx');
const input = new ort.Tensor('float32', gridData, [1, 3, gridSize, gridSize]);
const output = await session.run({ input });
const action = output.action.data[0];
```

Växla mellan "Browser Agent" och "Python Model (ONNX)" i Watch-läget och spara valet i `localStorage`.

## Vanliga frågor

**Renderingen hackar när jag kör flera miljöer.** Endast miljö `index 0` renderas i realtid för att undvika att pygame-fönster krockar. Övriga miljöer körs i bakgrunden.

**Kan jag köra utan rendering?** Ja, sätt miljön i silent mode via flaggan `--no-render` på träningsskripten. Miljön kommer då inte att öppna något fönster.

**Hur återupptar jag träning från en sparad modell?** Båda träningsskripten accepterar flaggan `--load <model_path>` för att återuppta träning.

Lycka till med träningen!
