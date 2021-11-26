## Инструменты

3 инструмента для проверки качества кода:

* линтер: pycodestyle
* линтер и security analisys tool: Bandit
* линтер для анализа исходников: prospector

### pycodestyle

Запускался со следующими настройками: 

```(shell)
pycodestyle --statistics -qq .
```

Команда запускает проверку на всех скриптовых .py файлах в текущей директории и проверяет их на соответствие pep8. Все ошибки в итоге аггрегируются и считаются по уникальности. 

Выдача команды:

```(shell)
34      E501 line too long (86 > 79 characters)
``` 

Нашлось 34 строки, превышающие рекомендованную длину.

### Bandit

Запускался со следующими настройками: 

```(shell)
bandit -r -l -i -v -x .git .
```

Проверка кодовой базы на уязвимости. Сообщается обо всех ошибках даже с низким уровнем уверенности и низким уровнем угрозы безопастности. Помимо этого выводятся все обойденные файлы и все пропущенные.

Вывод команды:

```(shell)
[main]	INFO	profile include tests: None
[main]	INFO	profile exclude tests: None
[main]	INFO	cli include tests: None
[main]	INFO	cli exclude tests: None
[main]	INFO	running on Python 3.6.9
Run started:2021-11-07 11:16:13.673687
Files in scope (5):
	./dataset.py (score: {SEVERITY: 0, CONFIDENCE: 0})
	./metrics.py (score: {SEVERITY: 0, CONFIDENCE: 0})
	./models.py (score: {SEVERITY: 0, CONFIDENCE: 0})
	./preprocessing.py (score: {SEVERITY: 0, CONFIDENCE: 0})
	./training.py (score: {SEVERITY: 3, CONFIDENCE: 10})
Files excluded (86):
	./.git/HEAD
	./.git/config
	./.git/description
	./.git/hooks/applypatch-msg.sample
	./.git/hooks/commit-msg.sample
	./.git/hooks/fsmonitor-watchman.sample
	./.git/hooks/post-update.sample
	./.git/hooks/pre-applypatch.sample
	./.git/hooks/pre-commit.sample
	./.git/hooks/pre-push.sample
	./.git/hooks/pre-rebase.sample
	./.git/hooks/pre-receive.sample
	./.git/hooks/prepare-commit-msg.sample
	./.git/hooks/update.sample
	./.git/index
	./.git/info/exclude
	./.git/logs/HEAD
	./.git/logs/refs/heads/master
	./.git/logs/refs/remotes/origin/HEAD
	./.git/objects/0a/9fa036fbac349e91f51c85b770de48aaecc777
	./.git/objects/0d/f57c093a70304743d5486b433a6a7526cea47c
	./.git/objects/14/10fdd0783f4ae952723785e1620cb504ac1275
	./.git/objects/16/ed91bac2ea5537562cb677607b75797963806d
	./.git/objects/18/739b8ae0d6b14c46d61e2802cb3c49f7c71cb7
	./.git/objects/19/edfc12a9c12fd341dcb5dbf9d4415bea4e0302
	./.git/objects/1a/4aa6a11996355beae3f1f49175a6cf916bdd11
	./.git/objects/23/39628bd09a6c2eee9b06b1f4973fd4814c8d0a
	./.git/objects/27/af943539911dc4f3f6765d9368215318f177f5
	./.git/objects/2b/158fca24dc34b81283ba0d277eb0e4b2db42f9
	./.git/objects/2b/a5dd9a9c3b329bb950da699bf549c921cae3ea
	./.git/objects/2c/30824b84f373b05f87ae397db5b8670d40c4d5
	./.git/objects/30/95cf11a582c14126a52097b9e0a2fbbfe5b4c5
	./.git/objects/31/cf1bf331e9346fc70acc37001e4bd8d67769e2
	./.git/objects/32/b1f2aa02e15a8d107aef3a61ca92ca3ec49bee
	./.git/objects/37/485a6bc8ee7b589f5e54b276bceeecf86018c2
	./.git/objects/37/9b2a18763ab15e2e4d742b96f7560cf05e7209
	./.git/objects/42/430002bbfd320d4f38d0c18ae564a8a6c5ca34
	./.git/objects/44/405c9b937ef250c777d443bcf4726e6081928c
	./.git/objects/4e/6ad742c6c1c9714cc463fe32c60e9b3f14d78f
	./.git/objects/4e/f96e4ef6b0045cdbc10704079cf30466599bec
	./.git/objects/53/7221450ac363d9b14402cab2a6e85dfa163321
	./.git/objects/56/d8ac4d75ba1fa2fd7dc53e31ae7e69c2eb7807
	./.git/objects/5b/2092fea9ba90fab868480d43d901606d312e06
	./.git/objects/61/8e0f5e8783e6cab1f35b048d6da5c6a2a2caf6
	./.git/objects/6e/20d200caebb6b1f644c3e98845ae8c80d347d7
	./.git/objects/76/bf48f6a7b4c8f299e37e64b6043bec954843e1
	./.git/objects/7a/9f73bfae1a7475b5b76392e5ba1df8322f5424
	./.git/objects/8b/ad381b5666ee4fa85d62a33f84ac160664f807
	./.git/objects/8d/9aceb54e31e4628d254b87629355fc9d17d925
	./.git/objects/8f/c9f128de51bbe14ed105f0454597431ac59abd
	./.git/objects/92/4973b26534117b98e29dfc69ef67313e161633
	./.git/objects/94/8f387586c679c92c9d9f70ee5a191dcaf80399
	./.git/objects/94/99113ff2a8f9487deff75679ee85ad07b145b9
	./.git/objects/96/807fdf70cbc2f6e0a7a9965b902e2ce3fb6586
	./.git/objects/a2/7b8f01797dbec92b1abd8d094f0f453799a9b4
	./.git/objects/a2/880ea9221e23881157b89492cc662566016c7c
	./.git/objects/a2/b90f88d7ef7a7319e549d091d655fb4a2dfea5
	./.git/objects/a8/5da1390ade0a437e24c82654472e03d9ff4168
	./.git/objects/b1/75aab761c85fa62e77baceffed70377fd2605a
	./.git/objects/b5/7092d4d7e80d6f179baff0a2d0406ca21c083a
	./.git/objects/b6/6e50b29e469c1a88f2a0fe1e5397741c4dc5c6
	./.git/objects/bb/7ba2c014f806080cd96ed59896eea50f62a1dd
	./.git/objects/d4/3779946db87b7dee8a47f72aa1a55a9f3b1d08
	./.git/objects/d4/7426a8cbac7cfdf57f7d09ea8088bb5835fa7a
	./.git/objects/d5/1ccd35e15bf51796e71fac553b05e4bd3edc34
	./.git/objects/da/3bba29449ae7ad90fd60f0f587bbe3ddacf845
	./.git/objects/df/7bed995a3a3dde6bb9cd7d31804244898fc204
	./.git/objects/df/e0ce85208f4b6df3e88113c4d02bbfccf08272
	./.git/objects/e5/5a51cada89a0677dcfe9f85de2391f20b88417
	./.git/objects/e8/97aae2da892fd904f6c185d4040efd770af778
	./.git/objects/eb/ba65361fa4e505ff9af4036f3ec0d72e5c0067
	./.git/objects/ec/326ca25fd9346e24c846a01a753b9bb1af057b
	./.git/objects/ed/3536fd119eb4fc75a0e45209f1928eedcd0a22
	./.git/objects/ed/f98e36c7b98476f80f375668219f3c973c9717
	./.git/objects/f0/d0e9f490f47b1a814b1dbb75b06708b8c2aeab
	./.git/objects/fa/1349a22d818519f80de114549977906027b8e5
	./.git/objects/fb/58ae103fa4f92b5f281d299da9e4f8b750d413
	./.git/objects/fd/b16605430f51d9e94b4f65f7a203504d4f9031
	./.git/packed-refs
	./.git/refs/heads/master
	./.git/refs/remotes/origin/HEAD
	./.gitignore
	./code_style_report.md
	./info/history.jpg
	./report.ipynb
	./requirements.txt

Test results:
>> Issue: [B403:blacklist] Consider possible security implications associated with pickle module.
   Severity: Low   Confidence: High
   Location: ./training.py:7
   More Info: https://bandit.readthedocs.io/en/latest/blacklists/blacklist_imports.html#b403-import-pickle
6	import matplotlib.pyplot as plt
7	import pickle
8	
9	from models import Generator, Discriminator, DiscriminatorLoss, GeneratorLoss

--------------------------------------------------

Code scanned:
	Total lines of code: 370
	Total lines skipped (#nosec): 0

Run metrics:
	Total issues (by severity):
		Undefined: 0.0
		Low: 1.0
		Medium: 0.0
		High: 0.0
	Total issues (by confidence):
		Undefined: 0.0
		Low: 0.0
		Medium: 0.0
		High: 1.0
Files skipped (0):
``` 

Найдена незначительная угроза связанная с безопасностью модуля pickle. Возможно в профессиональных кругах предпочитают проводить сериализацию моделей с помощью других инструментов, например protobuf.

### prospector

Использованная конфигурация:

```(shell)
prospector --strictness medium
```

Команда аггрегирует выводы нескольких линтеров проверяющих код на разные проблемы. Уровень строгости выбран средный, так как высокий уровень ругался даже на переменные с именами x и y, что на мой взгляд уже совсем бред.

Вывод команды:

```(shell)
Messages
========

dataset.py
  Line: 1
    pylint: import-error / Unable to import 'torch.utils.data'
  Line: 3
    pylint: import-error / Unable to import 'librosa'
  Line: 4
    pylint: import-error / Unable to import 'numpy'
  Line: 80
    pylint: import-error / Unable to import 'scipy.io.wavfile' (col 4)

metrics.py
  Line: 1
    pylint: import-error / Unable to import 'pesq'
  Line: 2
    pylint: import-error / Unable to import 'pystoi.stoi'

models.py
  Line: 1
    pylint: consider-using-from-import / Use 'from torch import nn' instead
    pylint: import-error / Unable to import 'torch.nn'
  Line: 2
    pylint: import-error / Unable to import 'torch'
  Line: 26
    pylint: unused-variable / Unused variable 'h' (col 11)

preprocessing.py
  Line: 1
    pylint: import-error / Unable to import 'librosa'
  Line: 2
    pylint: import-error / Unable to import 'scipy.signal'
  Line: 3
    pylint: import-error / Unable to import 'numpy'
  Line: 4
    pylint: import-error / Unable to import 'torch'
  Line: 36
    pylint: import-error / Unable to import 'matplotlib.pyplot' (col 4)
    pylint: unused-import / Unused matplotlib.pyplot imported as plt (col 4)
  Line: 37
    pylint: import-error / Unable to import 'librosa.display' (col 4)
  Line: 40
    pylint: import-error / Unable to import 'scipy.io.wavfile' (col 4)

training.py
  Line: 1
    pylint: import-error / Unable to import 'torch.utils.data'
  Line: 2
    pylint: import-error / Unable to import 'tqdm'
  Line: 3
    pylint: import-error / Unable to import 'numpy'
  Line: 4
    pylint: consider-using-from-import / Use 'from torch import optim' instead
    pylint: import-error / Unable to import 'torch.optim'
  Line: 5
    pylint: import-error / Unable to import 'torch'
  Line: 6
    pylint: import-error / Unable to import 'matplotlib.pyplot'
  Line: 12
    pylint: unused-import / Unused calculate_stoi imported from metrics
  Line: 19
    pylint: no-else-return / Unnecessary "else" after "return" (col 4)
  Line: 36
    pylint: too-many-locals / Too many local variables (19/15)
  Line: 42
    pylint: unused-variable / Unused variable 'x_phase' (col 20)
  Line: 81
    pylint: too-many-arguments / Too many arguments (6/5)
    pylint: too-many-locals / Too many local variables (48/15)
    pylint: too-many-statements / Too many statements (89/60)
  Line: 118
    pylint: unused-variable / Unused variable 'x_phase' (col 24)
  Line: 138
    pylint: unused-variable / Unused variable 's_phase' (col 25)
  Line: 169
    pep8: E501 / line too long (184 > 159 characters) (col 160)

Check Information
=================
         Started: 2021-11-07 14:26:16.134155
        Finished: 2021-11-07 14:26:18.019867
      Time Taken: 1.89 seconds
       Formatter: grouped
        Profiles: default, strictness_medium, strictness_high, strictness_veryhigh, no_doc_warnings, no_test_warnings, no_member_warnings
      Strictness: medium
  Libraries Used: 
       Tools Run: dodgy, mccabe, pep8, profile-validator, pyflakes, pylint
  Messages Found: 35

```

Найдено много ошибок с импортами, это ожидаемо, так как не было подготовленной виртуальной среды для проекта и поэтому многие из необходимых зависимостей не были поставлены. С остальными ошибками согласен, код может быть неоптимальный, но это неудивительно, кажется проект переписывался с какого-то китайского репозитория на втором питоне и на керасе и многие смысловые конструкции были заимствованы оттуда. 

## Статистика

Почти все инструменты жаловались на длинные строчки кода в некоторых файлах, сего таких строчек около 34 штук. Несколько импортированных но не использованных модулей и заведение неиспользуемых переменных.

## Грубые ошибки

В целом таких не нашлось, в конце концов код писался в идее, а она достаточно хорошо выполняет роль линтеров и прочих инструментов для анализа кода. Открытием для меня стало то что возможно использовать pickle не самое безопасное решение, не знаю с чем это может быть связано.
