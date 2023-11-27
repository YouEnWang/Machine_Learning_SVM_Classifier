程式執行方式
 - 從main function設定SVM參數，並呼叫SVM function
 - SVM(initial_data, kernel, C, P=None, sigma=None)

kernel類型設定
 - 從kernel_type中選擇要使用的kernel
 - linear SVM雖然沒有kernel，但為了增加readability以及方便程式執行，仍為其命名為"Linear"