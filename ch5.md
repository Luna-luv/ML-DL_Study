# Ch.5 트리 알고리즘
## 1. 결정 트리
## cf
```python
df.describe()
```
- 시각화가 아니라 describe 를 활용해서 스케일 확인하고 scaler 사용 여부를 결정해도 됨

```python
data = df.[['행1', '행2', '행3', ...]].to_numpy()
target = df['타겟변수'].to_numpy()
```
- Scikit-learn 같은 ML 라이브러리들이 NumPy 배열을 입력받도록 설계되어 있기 때문에 numpy 배열로 변경 必
```python
random state = 00
```
- random state 숫자가 같을 경우 동일한 랜덤 데이터로 훈련
- 실전에서는 필요하지 않지만, 연습 또는 코드 공유 시에 같은 숫자로 하면 동일한 점수를 얻을 수 있음