import numpy as np
from cnn.util import im2col

def test_im2col():
    # 1. データ準備: 1から16までの数字が並んだ画像 (1, 1, 4, 4)
    # [[ 1,  2,  3,  4],
    #  [ 5,  6,  7,  8],
    #  [ 9, 10, 11, 12],
    #  [13, 14, 15, 16]]
    x = np.arange(1, 17).reshape(1, 1, 4, 4)

    # 2. im2col実行
    # フィルターサイズ: 2x2
    # ストライド: 2 (2マスずつ飛ぶ)
    # パディング: 0
    col = im2col(x, filter_h=2, filter_w=2, stride=2, pad=0)

    print("Input shape:", x.shape)
    print("Output shape:", col.shape)
    print("\nResult (im2col output):")
    print(col)

    # 3. 正解チェック
    # ストライド2なので、以下の4つのブロックが切り出されるはず
    # [1, 2, 5, 6]   (左上)
    # [3, 4, 7, 8]   (右上)
    # [9, 10, 13, 14] (左下)
    # [11, 12, 15, 16] (右下)
    
    expected = np.array([
        [1, 2, 5, 6],
        [3, 4, 7, 8],
        [9, 10, 13, 14],
        [11, 12, 15, 16]
    ])

    np.testing.assert_array_equal(col, expected)

    if np.array_equal(col, expected):
        print("\n✅ Test Passed! im2col is working correctly.")
    else:
        print("\n❌ Test Failed!")
        print("Expected:\n", expected)

if __name__ == "__main__":
    test_im2col()
