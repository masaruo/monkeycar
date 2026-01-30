import numpy as np
from cnn.util import im2col, col2im

def test_col2im():
    print("Testing col2im...")
    
    # 1. データ準備: 1〜16の画像
    input_img = np.arange(1, 17).reshape(1, 1, 4, 4)
    
    # 2. 一度バラバラにする (im2col)
    # フィルター2x2, ストライド2 (重なりなし)
    col = im2col(input_img, filter_h=2, filter_w=2, stride=2, pad=0)
    
    # 3. 元に戻す (col2im)
    # 期待値: input_img と全く同じになるはず
    output_img = col2im(col, input_shape=input_img.shape, 
                        filter_h=2, filter_w=2, stride=2, pad=0)

    print("\nOriginal Input:\n", input_img)
    print("\nReconstructed Output:\n", output_img)

    np.testing.assert_array_equal(input_img, output_img)

    if np.array_equal(input_img, output_img):
        print("\n✅ Test Passed! col2im restored the image perfectly.")
    else:
        print("\n❌ Test Failed! Values do not match.")

if __name__ == "__main__":
    test_col2im()
