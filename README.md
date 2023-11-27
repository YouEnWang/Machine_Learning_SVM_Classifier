# 目標
利用鳶尾花資料(Iris data set)來訓練Support Vector Machine (SVM)，並分別比較線性與非線性SVM所訓練的超平面(hyperplane)有何差異。

# 資料描述
1. 安德森鳶尾花卉數據集(Anderson's Iris data set)為機器學習領域中，常被用來驗證演算法效能的資料庫。數據庫中包含三種不同鳶尾花標籤(Label)：山鳶尾(Setosa)、變色鳶尾(Versicolor)以及維吉尼亞鳶尾(Virginica)，且每種鳶尾花皆有50筆樣本。而每筆樣本以四種屬性作為特徵(單位：cm)：花萼長度(Sepal length)、花萼寬度(Sepal width)、花瓣長度(Petal length)以及花瓣寬度(Petal width)等四種屬性作為定量分析之數據。
2. 讀取鳶尾花資料後會產生150×5的陣列，其中第5行為資料的類別標籤。

# 作業內容
1. Linear SVM (Initialization: C=1)
2. RBF kernel-based SVM (Initialization: C=10, sigma=5)
3. Polynomial kernel-based SVM (Initialization: C=10, p=1)
4. Discussion and results presenting

# 程式執行方式
 - 從main function設定SVM參數，並呼叫SVM function
 - SVM(initial_data, kernel, C, P=None, sigma=None)

## kernel類型設定
 - 從kernel_type中選擇要使用的kernel
 - linear SVM雖然沒有kernel，但為了增加readability以及方便程式執行，仍為其命名為"Linear"

# 討論
1. Linear SVM與kernel-based SVM所訓練的hyperplane有何差異？
	- 兩者主要區別在於它們對超平面形狀和特點的處理方式。前者適用於線性可分數據；而後者則可處理更複雜的非線性數據分佈，因為它可以在高維度空間中找出hyperplane。kernel-based SVM鈄過選擇適當的核函數，能夠適應各種數據分佈，包括非線性和高度不可分的情況。

2. 隨著kernel parameter的改變，RBF kernel與polynomial kernel所訓練的hyperplane可能有什麼變化？其與分類率的變化有何關聯？請嘗試解釋之。
	- RBF kernel：隨著sigma增加，RBF kernel的hyperplane會更貼近支持向量，可能導致過擬合(overfitting)。
	- polynomial kernel：隨著poly增加，polynomial kernel的hyperplane會更複雜，能夠捕捉更多的數據特徵，但同樣可能導致過擬合(overfitting)。
	- 合適的kernel參數能夠提高分類率，而過度調整kernel參數可能會導致過擬合，並對測試數據的性能產生負面影響，造成分類率下降。

3. 設定kernel parameter時，是否有方法避免hyperplane過度擬合(overfitting)的現象發生？若有請詳細討論。
   - 交叉驗證：通過k-fold 交叉驗證，可以測試不同kernel參數對不同訓練集與測試集組合的分類效果，以確保模型有好的泛化性能。
   - 網格搜索：可以自動化找到最佳參數設置，以此來減少過度擬合的風險。
   - 正規化：調整正規化參數，可以控制模型的複雜度，降低過擬合發生的可能性。

4. 結果分析
   - alpha: 觀察數據可以發現，分類率較高的實驗會有較多training data的alpha值為0。較少的alpha超出0 ~ C這個範圍，亦即其support vectors數量較少。雖然support vectors的數量並無法直接作為評估模型性能的標準，但較少的support vectors可以降低過度擬合發生的機率，因為這些support vectors代表最關鍵的訓練樣本點，而不是所有訓練樣本。然而從poly = 5的實驗觀察出，在高維度空間中，SVM似乎需要較多的支持向量才能實現良好的分類性能。可能的原因除了高度過擬合外，還可能是因為在高維度空間中，support vectors的選擇容易受到噪聲或微小變化的影響，導致模型的不穩定性。

   - bias: bias表示超平面的截距，調整bias可以確保超平面在正確的位置分隔不同的類別，從而實現良好的分類結果。觀察分類率與bias並無發現直接的相關性，但在撰寫程式時我發現bias的絕對值如果大至超過200，有很大機率代表程式有某個段落設計錯誤，最後得出的分類率會為50 %，表示分類的結果完全偏向其中一個類別。

# 心得
設計SVM演算法是一件難度很高的事情，我反覆看了上課教材數次才理解bias跟Decision rule公式中，summation指定的對象與範圍。在設計Decision rule時，我起初是以先計算完一個row再將其append到kernel_set內部，導致我的row跟column顛倒，不管嘗試幾次都得出CR = 50 %的結果。後來透過觀察一個一個的element才發現此狀況，而後為了避免此狀況，我改成預先創立一個全為0的matrix (指的是kernel_set)，並透過每次的loop中更動每個element，才有效避免搞混row跟column的問題。光是發現問題的過程就消耗我兩天的時間，讓我記取了很大的教訓。
