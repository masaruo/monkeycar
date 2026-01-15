import unittest
import os
import keras
from trainer import Trainer  # ファイル名が trainer.py であると仮定

class TestTrainerIntegration(unittest.TestCase):
    """
    モックを使用せず、実際のクラスとライブラリを用いて
    学習パイプライン全体の実行可能性を検証する。
    """

    def setUp(self):
        """テスト実行前の環境確認"""
        self.trainer = Trainer()
        self.output_path = "output/model.keras"
        
        # 出力ディレクトリの担保
        os.makedirs("output", exist_ok=True)

    def test_train_flow(self):
        """
        trainメソッドを実行し、正常にモデルが保存されるまでを検証する。
        注: Loader.load_sessions() が適切なデータを返す環境であることを前提とする。
        """
        # 1. 学習の実行
        # 実データを用いるため、Loaderの仕様に準じたデータが配置されている必要がある。
        try:
            self.trainer.train()
        except Exception as e:
            self.fail(f"Trainer.train() が予期せぬ例外をスローしました: {e}")

        # 2. 成果物の検証
        # ファイルが生成されているか
        self.assertTrue(os.path.exists(self.output_path), "モデルファイルが保存されていません。")

        # 3. モデルの妥当性検証
        # 実際にロードしてKerasモデルとして機能するか確認
        try:
            model = keras.models.load_model(self.output_path)
            self.assertIsInstance(model, keras.Model)
            
            # 入力シェイプの不整合がないか等の簡易チェック
            # (Loaderから渡されたデータの形状と一致しているか)
            self.assertIsNotNone(model.input_shape)
        except Exception as e:
            self.fail(f"保存されたモデルのロードに失敗しました: {e}")

if __name__ == '__main__':
    unittest.main()
