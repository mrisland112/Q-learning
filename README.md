# Q-learning
- 傳統的 Q-learning 是基於 RL 的表計算，不涉及大規模的矩陣運算，因此没有直接的方法，可以將其在 GPU 上進行處理 (訓練時間約 3 分鐘)
- 用於解決在给定環境下的馬可夫決策過程（Markov Decision Process，MDP）中的最佳化問題
- 可以拓展成 DQN (結合神經網路的形式)，就可以使用 GPU 進行運算，進而去提高模型的訓練和預測速度

## 使用技術
- OpenAI Gym 建立遊戲環境
- 建立 Q-learning 演算法

## 目的
- 利用 Reinforcement learning 的方式，去訓練模型得到最大化獎勵，也就是讓模型自行去調整滑動台的速度，使得木條可以保持物理上的力學平衡狀態
- 每局的最大 total rewards 為 200

## 設立環境
```
pip install gym==0.24.0
pip install numpy
pip install matplotlib
pip install gym[classic_control]        #如果使用低於python 3.9.x的版本需要額外使用此指令
pip install pygame
```

## 訓練過程
- 在初始訓練時，平均 total reward 偏低，代表模型處在一個不穩定狀態，也就是滑動台的速度沒辦法追上木條墜落的速度，很容易產生力學不平衡的問題

![pygame window 2023-06-06 10-35-17 (online-video-cutter com)](https://github.com/mrisland112/Q-learning/assets/28065019/313ec25d-e50e-46f9-9a37-ca545edd483e)

![train_init](https://github.com/mrisland112/Q-learning/assets/28065019/2763605f-97f8-4cb9-8b24-894eac6ebb55)

## 訓練結果
- 可以發現當訓練 episode 達到 15000 次時，平均 total reward 達到收斂

![train](https://github.com/mrisland112/Q-learning/assets/28065019/de7fe85b-fe65-4677-ab12-078380757fcb)


## 儲存訓練結果
- pole.py 執行完會儲存 qtable.npy，在後續測試我們會使用到這個檔案

## 測試
- 可以發現訓練好的模型可以讓滑動台維持高頻率震動，達成讓木條不墜落的任務

![pygame window 2023-06-07 22-42-29 (online-video-cutter com)](https://github.com/mrisland112/Q-learning/assets/28065019/8188f98c-86ce-4489-ae66-30ac6f16c596)



