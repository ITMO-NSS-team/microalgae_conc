import sys
import os
import math
import pandas as pd
from pathlib import Path
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                               QStackedWidget, QFrame, QGridLayout, QMessageBox,
                               QProgressDialog, QFileDialog, QComboBox)
from PySide6.QtCore import Qt, QThread, Signal
import cv2
import traceback


from CellCounter.Segmentator import detect_cells, visualize_circles
from CellCounter.ConcentrationCalculator import calculate_concentration
from CellCounter.VolumeCalculator import calc_volume


# КОНСТАНТЫ СТИЛЯ
COLOR_BG_MAIN = "#734d4d"
COLOR_CARD_BG = "#5a3e3e"
COLOR_ACCENT = "#8c6161"
COLOR_INPUT_BG = "#8e6b6b"
COLOR_TEXT = "#ffffff"
COLOR_BTN_RUN = "#4a3232"
COLOR_BTN_HOVER = "#5e4040"

DEFAULT_VALUES = {
    "коэф. сглаживания": "3",
    "мин. расстояние (пикс.)": "10",
    "мин. радиус (пикс.)": "3",
    "макс. радиус (пикс.)": "20",
    "верхний порог": "20",
    "нижний порог": "30",
    "контрастность": "нет",
    "обозначение": "нет",
    "средний радиус": "нет"
}

COMBO_OPTIONS = {
    "контрастность": ["нет", "да"],
    "обозначение": ["нет", "да"],
    "средний радиус": ["нет", "да"]
}

MANDATORY_FIELDS = ["размер сетки (мм)", "глубина камеры (мм)", "коэф. разбавления"]

MANDATORY_DEFAULTS = {
    "размер сетки (мм)": "0.05",
    "глубина камеры (мм)": "0.1",
    "коэф. разбавления": "1"
}

STYLE = f"""
    QMainWindow {{ 
        background-color: {COLOR_BG_MAIN}; 
    }}
    QWidget {{ 
        color: {COLOR_TEXT}; 
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif; 
        font-size: 16px; 
    }}
    
    QLineEdit, QComboBox {{
        background-color: {COLOR_INPUT_BG};
        border-radius: 8px;
        padding: 8px 12px;
        color: white;
        border: 1px solid rgba(255,255,255,0.3);
        font-size: 15px;
    }}
    QLineEdit:focus, QComboBox:focus {{
        border: 1px solid white;
        background-color: {COLOR_ACCENT};
    }}
    QComboBox::drop-down {{
        border: 0px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {COLOR_CARD_BG};
        color: white;
        selection-background-color: {COLOR_ACCENT};
    }}
    
    QPushButton {{
        background-color: {COLOR_CARD_BG};
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        border: none;
    }}
    QPushButton:hover {{ 
        background-color: {COLOR_ACCENT}; 
        cursor: pointer;
    }}
    QPushButton:pressed {{
        background-color: {COLOR_BTN_RUN};
    }}
    
    #BigInput {{ 
        min-height: 45px; 
        border-radius: 12px; 
        padding-left: 15px; 
        font-size: 16px; 
        background-color: rgba(255,255,255,0.15);
        border: none;
    }}
    
    #RunBtn {{ 
        background-color: {COLOR_BTN_RUN}; 
        font-size: 18px; 
        font-weight: bold;
        letter-spacing: 1px;
        min-height: 60px; 
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.2);
        margin-top: 10px;
    }}
    #RunBtn:hover {{
        background-color: {COLOR_ACCENT};
        border: 1px solid white;
    }}
    
    #Card {{ 
        background-color: {COLOR_CARD_BG}; 
        border-radius: 20px; 
        padding: 25px; 
    }}
    
    #BrowseBtn {{
        background-color: rgba(255,255,255,0.1);
        font-size: 20px;
        border-radius: 12px;
    }}
    
    #TitleLabel {{
        font-size: 48px; 
        font-weight: 800;
        color: white;
    }}
    
    #DescLabel {{
        font-size: 18px;
        color: rgba(255,255,255,0.8);
        font-style: italic;
    }}
    
    #SectionTitle {{
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 10px;
        color: #ffdddd;
    }}
"""

DIALOG_STYLE = f"""
    QDialog {{
        background-color: {COLOR_BG_MAIN};
    }}
    QDialog QLabel {{
        color: {COLOR_TEXT};
        font-size: 14px;
    }}
    QDialog QPushButton {{
        background-color: {COLOR_CARD_BG};
        color: {COLOR_TEXT};
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        min-width: 100px;
    }}
    QDialog QPushButton:hover {{
        background-color: {COLOR_ACCENT};
    }}
    QMessageBox {{
        background-color: {COLOR_BG_MAIN};
    }}
    QMessageBox QLabel {{
        color: {COLOR_TEXT};
        font-size: 14px;
        min-width: 300px;
    }}
    QMessageBox QPushButton {{
        background-color: {COLOR_CARD_BG};
        color: {COLOR_TEXT};
        border-radius: 6px;
        padding: 6px 16px;
        font-weight: 600;
        min-width: 80px;
    }}
    QMessageBox QPushButton:hover {{
        background-color: {COLOR_ACCENT};
    }}
"""

class ProcessingThread(QThread):
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(str)
    
    def __init__(self, image_path, calibration_folder, params):
        super().__init__()
        self.image_path = image_path
        self.calibration_folder = calibration_folder
        self.params = params
        
    def run(self):
        try:
            output_dir = Path(self.image_path).parent / "results"
            output_dir.mkdir(exist_ok=True)
            
            self.progress.emit("Загрузка изображения...")
            image = cv2.imread(self.image_path)
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение: {self.image_path}")
            
            self.progress.emit("Обнаружение клеток...")

            increase_channel = None
            if self.params.get('контрастность', 'нет').lower() == 'да':
                increase_channel = 1

            circles = detect_cells(
                image,
                increase_channel=increase_channel,
                minDist=int(self.params.get('мин. расстояние (пикс.)', 10)),
                minRadius=int(self.params.get('мин. радиус (пикс.)', 3)),
                maxRadius=int(self.params.get('макс. радиус (пикс.)', 20)),
                param2=int(self.params.get('верхний порог', 20)),
                blur_kernel=int(self.params.get('коэф. сглаживания', 3))
            )
            
            if circles.shape[1] == 0:
                raise ValueError("Клетки не обнаружены. Попробуйте изменить параметры порогов или радиуса.")
            
            cell_count = circles.shape[1]
            self.progress.emit(f"Обнаружено клеток: {cell_count}")
            
            self.progress.emit("Расчет объема и калибровка...")
            grid_size = float(self.params['размер сетки (мм)'])
            depth = float(self.params['глубина камеры (мм)'])
            P_h, P_w = image.shape[:2]
            
            calib_plot_path = output_dir / f"calibration_stats_{Path(self.image_path).stem}.png"
            
            v_img = calc_volume(
                imgs_path=self.calibration_folder,
                l=grid_size,
                h=depth,
                P_h=P_h,
                P_w=P_w,
                plot_stats_path=str(calib_plot_path) 
            )

            # Вычисление коэффициента масштаба
            s_squared = v_img / (P_h * P_w * depth)
            mm_per_pixel = math.sqrt(s_squared)
            
            self.progress.emit("Расчет концентрации...")
            dilution = float(self.params['коэф. разбавления'])
            concentration = calculate_concentration(cell_count, dilution, v_img)
            
            self.progress.emit("Сохранение результатов...")
            
            output_path = output_dir / f"result_{Path(self.image_path).stem}.png"
            txt_output_path = output_dir / f"result_{Path(self.image_path).stem}.txt"
            
            show_radius = self.params.get('средний радиус', 'нет').lower() in ['да', 'yes']
            show_annotation = self.params.get('обозначение', 'нет').lower() in ['да', 'yes']
            
            visualize_circles(image, circles, save_path=str(output_path),
                              annotate=show_annotation, ext_title=show_radius)
            
            self.save_results_to_txt(txt_output_path, cell_count, v_img, concentration,
                                     self.image_path, self.calibration_folder, self.params, mm_per_pixel)
            
            result = {
                'cell_count': cell_count,
                'concentration': concentration,
                'volume': v_img,
                'scale_factor': mm_per_pixel,
                'output_path': str(output_path),
                'txt_path': str(txt_output_path),
                'output_dir': str(output_dir),
                'calib_plot': str(calib_plot_path)
            }
            self.finished.emit(result)
            
        except Exception as e:
            error_log = traceback.format_exc()
            self.error.emit(error_log)
    
    def save_results_to_txt(self, filepath, cell_count, volume, concentration, 
                           image_path, calib_folder, params, scale_factor):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Дата и время анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("ВХОДНЫЕ ДАННЫЕ\n")
            f.write(f"Изображение для анализа: {image_path}\n")
            f.write(f"Папка калибровки: {calib_folder}\n\n")
            f.write("ПАРАМЕТРЫ ОБРАБОТКИ\n")
            for k, v in params.items():
                f.write(f"{k}: {v}\n")
            
            f.write(f"\nКАЛИБРОВКА\n")
            f.write(f"Коэффициент масштаба: {scale_factor:.6e} мм/пиксель\n")

            f.write("\nРЕЗУЛЬТАТЫ АНАЛИЗА\n")
            f.write(f"Обнаружено клеток: {cell_count}\n")
            f.write(f"Объем изображения: {volume:.6f} мм³\n")
            f.write(f"Концентрация: {concentration} клеток/мл\n")

class MainScreen(QWidget):
    def __init__(self, nav, run_logic, browse_calibration, browse_samples):
        super().__init__()
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(50, 40, 50, 40)
        main_layout.setSpacing(20)
        
        top_bar = QHBoxLayout()
        top_bar.addStretch()
        btn_settings = QPushButton("Расширенные настройки ⚙️")
        btn_settings.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_settings.clicked.connect(lambda: nav(1))
        top_bar.addWidget(btn_settings)
        main_layout.addLayout(top_bar)

        main_layout.addStretch(1)
        title = QLabel("microalgae_analyzer")
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        desc = QLabel("Автоматический подсчет клеток и расчет концентрации\nмикроводорослей по изображениям с камеры Горяева")
        desc.setObjectName("DescLabel")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(desc)
        
        main_layout.addStretch(1)

        form_container = QWidget()
        form_layout = QVBoxLayout(form_container)
        form_layout.setSpacing(15)

        # ===== INPUT 1: CALIBRATION FOLDER (with multiple grid images) =====
        lbl_calib = QLabel("📊 Папка с калибровочными изображениями сетки (минимум 5 фото):")
        lbl_calib.setStyleSheet("font-weight: bold; color: #ffaaaa;")
        form_layout.addWidget(lbl_calib)

        row_calib = QHBoxLayout()
        self.path_calib = QLineEdit()
        self.path_calib.setObjectName("BigInput")
        self.path_calib.setPlaceholderText("Выберите папку с фото сетки Горяева...")
        btn_browse_c = QPushButton("📂")
        btn_browse_c.setObjectName("BrowseBtn")
        btn_browse_c.setFixedSize(50, 45)
        btn_browse_c.clicked.connect(lambda: browse_calibration(self.path_calib))
        row_calib.addWidget(self.path_calib)
        row_calib.addWidget(btn_browse_c)
        form_layout.addLayout(row_calib)

        # Optional: show calibration preview button
        btn_preview = QPushButton("Предпросмотр калибровки")
        btn_preview.setStyleSheet("font-size: 14px; background-color: #5a3e3e;")
        btn_preview.clicked.connect(self.preview_calibration)
        form_layout.addWidget(btn_preview, alignment=Qt.AlignmentFlag.AlignRight)

        form_layout.addSpacing(20)

        # ===== INPUT 2: SAMPLES FOLDER (with cell images to analyze) =====
        lbl_samples = QLabel("🔬 Папка с фотографиями микроводорослей для анализа:")
        lbl_samples.setStyleSheet("font-weight: bold;")
        form_layout.addWidget(lbl_samples)

        row_samples = QHBoxLayout()
        self.path_samples = QLineEdit()
        self.path_samples.setObjectName("BigInput")
        self.path_samples.setPlaceholderText("Выберите папку с фото клеток...")
        btn_browse_s = QPushButton("📂")
        btn_browse_s.setObjectName("BrowseBtn")
        btn_browse_s.setFixedSize(50, 45)
        btn_browse_s.clicked.connect(lambda: browse_samples(self.path_samples))
        row_samples.addWidget(self.path_samples)
        row_samples.addWidget(btn_browse_s)
        form_layout.addLayout(row_samples)

        # Info about output folder (will be created inside samples folder)
        info_label = QLabel("✓ Результаты будут сохранены в папку 'segmented' внутри папки с образцами")
        info_label.setStyleSheet("color: #aaffaa; font-size: 14px; padding: 5px;")
        info_label.setWordWrap(True)
        form_layout.addWidget(info_label)

        main_layout.addWidget(form_container)
        main_layout.addSpacing(20)

        btn_run = QPushButton("ЗАПУСТИТЬ ОБРАБОТКУ")
        btn_run.setObjectName("RunBtn")
        btn_run.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_run.clicked.connect(run_logic)
        main_layout.addWidget(btn_run)

        main_layout.addStretch(2)
        
        footer = QLabel("AIST 2025")
        footer.setStyleSheet("color: rgba(255,255,255,0.4); font-size: 14px;")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(footer)

    def preview_calibration(self):
        """Show calibration statistics and plot"""
        calib_folder = self.path_calib.text().strip()
        if not calib_folder or not os.path.exists(calib_folder):
            QMessageBox.warning(self, "Внимание", "Сначала выберите папку с калибровкой")
            return

        # Get images
        images = [f for f in os.listdir(calib_folder)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]

        if len(images) < 5:
            QMessageBox.warning(self, "Недостаточно изображений",
                                f"Найдено {len(images)} изображений.\nНужно минимум 5 для надежной калибровки.")
            return

        try:
            # Get main window reference
            main_window = self.window()

            # Get parameters from settings
            grid_size = float(main_window.sett_scr.inputs['размер сетки (мм)'].text())
            depth = float(main_window.sett_scr.inputs['глубина камеры (мм)'].text())

            # Get first sample image to get dimensions (if samples folder is selected)
            samples_folder = self.path_samples.text().strip()
            if samples_folder and os.path.exists(samples_folder):
                sample_images = [f for f in os.listdir(samples_folder)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
                                 and f != "segmented"]
                if sample_images:
                    first_img = cv2.imread(os.path.join(samples_folder, sample_images[0]))
                    if first_img is not None:
                        P_h, P_w = first_img.shape[:2]
                    else:
                        # Default dimensions if can't load
                        P_h, P_w = 1000, 1000
                else:
                    P_h, P_w = 1000, 1000
            else:
                P_h, P_w = 1000, 1000

            # Create temporary file for plot
            temp_plot = os.path.join(os.path.dirname(calib_folder), "temp_calib_preview.png")

            # Calculate volume (this will generate the plot)
            v_img = calc_volume(
                imgs_path=calib_folder,
                l=grid_size,
                h=depth,
                P_h=P_h,
                P_w=P_w,
                plot_stats_path=temp_plot
            )

            # Calculate scale factor
            s_squared = v_img / (P_h * P_w * depth)
            mm_per_pixel = math.sqrt(s_squared)

            # Show the plot in a new window
            if os.path.exists(temp_plot):
                from PySide6.QtGui import QPixmap
                from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel

                dialog = QDialog(self)
                dialog.setWindowTitle("Предпросмотр калибровки")
                dialog.resize(900, 600)
                # Стили наследуются от глобальных, не нужно устанавливать заново

                layout = QVBoxLayout(dialog)

                # Create label with image
                label = QLabel()
                pixmap = QPixmap(temp_plot)
                scaled_pixmap = pixmap.scaled(880, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(scaled_pixmap)
                label.setAlignment(Qt.AlignCenter)
                layout.addWidget(label)

                # Add info label
                info = QLabel(
                    f"Калибровочных изображений: {len(images)}\n"
                    f"Объем изображения: {v_img:.6f} мм³\n"
                    f"Масштаб: {mm_per_pixel:.6e} мм/пиксель\n\n"
                    f"Для более точной калибровки используйте минимум 5 изображений\n"
                    f"в разных участках камеры."
                )
                info.setAlignment(Qt.AlignCenter)
                layout.addWidget(info)

                # Add close button
                btn_close = QPushButton("Закрыть")
                btn_close.clicked.connect(dialog.accept)
                btn_close.setFixedWidth(200)
                layout.addWidget(btn_close, alignment=Qt.AlignCenter)

                dialog.exec()

                # Clean up temp file
                try:
                    os.remove(temp_plot)
                except:
                    pass
            else:
                QMessageBox.information(self, "Калибровка",
                                        f"Калибровочных изображений: {len(images)}\n"
                                        f"Объем изображения: {v_img:.6f} мм³\n"
                                        f"Масштаб: {mm_per_pixel:.6e} мм/пиксель")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось выполнить калибровку:\n{str(e)}")

PARAM_DESCRIPTIONS = {
    "коэф. сглаживания": "Размер ядра для медианного размытия (3, 5, 7...). Меньшие значения сохраняют больше деталей.",
    "мин. расстояние (пикс.)": "Минимальное расстояние между центрами обнаруженных кругов.",
    "мин. радиус (пикс.)": "Минимальный радиус клетки для обнаружения в пикселях.",
    "макс. радиус (пикс.)": "Максимальный радиус клетки для обнаружения в пикселях.",
    "верхний порог": "Чувствительность детектора (меньше = больше ложных срабатываний).",
    "нижний порог": "Порог детектора границ Canny (обычно оставляют 30).",
    "контрастность": "Улучшение контраста в синем канале (использовать только для тусклых изображений).",
    "обозначение": "Нумеровать ли обнаруженные круги.",
    "средний радиус": "Показывать ли расширенный заголовок со статистикой."
}

class SettingsScreen(QWidget):
    def __init__(self, nav):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        
        header = QHBoxLayout()
        btn_back = QPushButton("← Назад")
        btn_back.setFixedWidth(120)
        btn_back.clicked.connect(lambda: nav(0))
        header.addWidget(btn_back)
        header.addStretch()
        layout.addLayout(header)
        
        layout.addSpacing(20)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(30)
        self.inputs = {}

        # --- ЛЕВАЯ КАРТОЧКА ---
        left_card = QFrame()
        left_card.setObjectName("Card")
        left_layout = QVBoxLayout(left_card)
        left_layout.setSpacing(15)
        
        l_title = QLabel("Обязательные параметры")
        l_title.setObjectName("SectionTitle")
        l_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(l_title)
        left_layout.addWidget(QLabel("(введите значения вручную)"), alignment=Qt.AlignmentFlag.AlignCenter)
        left_layout.addSpacing(10)

        for name, default_val in MANDATORY_DEFAULTS.items():
            row_w = QWidget()
            row = QHBoxLayout(row_w)
            row.setContentsMargins(0,0,0,0)
            
            lbl = QLabel(name)
            lbl.setWordWrap(True)
            
            edit = QLineEdit()
            edit.setFixedWidth(140)
            edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
            edit.setText(default_val)
            edit.setPlaceholderText(default_val)
            
            edit.editingFinished.connect(
                lambda e=edit, d=default_val: e.setText(d) if not e.text().strip() else None
            )
            
            self.inputs[name] = edit
            row.addWidget(lbl, 1)
            row.addWidget(edit, 0)
            left_layout.addWidget(row_w)
        
        left_layout.addStretch()
        
        btn_def = QPushButton('Посмотреть "по умолчанию" ➜')
        btn_def.setStyleSheet(f"background-color: {COLOR_BG_MAIN};")
        btn_def.clicked.connect(lambda: nav(2))
        left_layout.addWidget(btn_def)

        # --- ПРАВАЯ КАРТОЧКА ---
        right_card = QFrame()
        right_card.setObjectName("Card")
        right_layout = QVBoxLayout(right_card)
        right_layout.setSpacing(12)

        r_title = QLabel("Дополнительные настройки")
        r_title.setObjectName("SectionTitle")
        r_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(r_title)
        
        hint_label = QLabel("(наведите на название для справки)")
        hint_label.setStyleSheet("font-size: 13px; color: rgba(255,255,255,0.6);")
        right_layout.addWidget(hint_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        right_layout.addSpacing(10)
        
        for name, default_val in DEFAULT_VALUES.items():
            row_w = QWidget()
            row = QHBoxLayout(row_w)
            row.setContentsMargins(0,0,0,0)
            
            lbl = QLabel(f"{name} ℹ️")
            lbl.setWordWrap(True)
            
            description = PARAM_DESCRIPTIONS.get(name, "Нет описания.")
            lbl.setToolTip(description)
            lbl.setCursor(Qt.CursorShape.WhatsThisCursor)
            
            if name in COMBO_OPTIONS:
                edit = QComboBox()
                edit.setFixedWidth(140)
                edit.addItems(COMBO_OPTIONS[name])
                idx = edit.findText(default_val)
                if idx >= 0:
                    edit.setCurrentIndex(idx)
            else:
                edit = QLineEdit()
                edit.setFixedWidth(140)
                edit.setPlaceholderText(str(default_val))
                edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            edit.setToolTip(description)
            
            self.inputs[name] = edit
            row.addWidget(lbl, 1)
            row.addWidget(edit, 0)
            right_layout.addWidget(row_w)

        right_layout.addStretch()

        content_layout.addWidget(left_card, 1)
        content_layout.addWidget(right_card, 1)
        
        layout.addLayout(content_layout)

class DefaultScreen(QWidget):
    def __init__(self, nav):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 20)
        
        header = QHBoxLayout()
        btn_back = QPushButton("← Назад к настройкам")
        btn_back.setFixedWidth(200)
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(lambda: nav(1))
        header.addWidget(btn_back)
        header.addStretch()
        layout.addLayout(header)

        layout.addStretch(1)

        card_h_layout = QHBoxLayout()
        card_h_layout.addStretch(1)

        card = QFrame()
        card.setObjectName("Card")
        card.setMinimumWidth(700)
        card.setMaximumWidth(900)
        
        grid = QGridLayout(card)
        grid.setContentsMargins(40, 25, 40, 25) 
        grid.setHorizontalSpacing(50)
        grid.setVerticalSpacing(6)

        title = QLabel("Параметры по умолчанию")
        title.setObjectName("SectionTitle")
        title.setStyleSheet("font-size: 22px; font-weight: bold; margin-bottom: 10px; color: #fff;")
        grid.addWidget(title, 0, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)

        all_params = {**MANDATORY_DEFAULTS, **DEFAULT_VALUES}
        
        row = 1
        for k, v in all_params.items():
            k_label = QLabel(k)
            k_label.setStyleSheet("color: #ddd; font-size: 15px; background: transparent;")
            
            v_label = QLabel(str(v))
            v_label.setStyleSheet("color: #fff; font-size: 17px; font-weight: bold; background: transparent;")
            v_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            
            grid.addWidget(k_label, row, 0)
            grid.addWidget(v_label, row, 1)
            
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFixedHeight(1)
            line.setStyleSheet("background-color: rgba(255,255,255,0.1); border: none;")
            grid.addWidget(line, row+1, 0, 1, 2)
            
            grid.setRowMinimumHeight(row, 28) 
            row += 2

        info_l = QLabel("Канал контраста (авто)")
        info_l.setStyleSheet("color: #ddd; font-size: 15px;")
        info_v = QLabel("0 (Blue channel)")
        info_v.setStyleSheet("color: #fff; font-size: 17px; font-weight: bold;")
        info_v.setAlignment(Qt.AlignmentFlag.AlignRight)
        grid.addWidget(info_l, row, 0)
        grid.addWidget(info_v, row, 1)
        grid.setRowMinimumHeight(row, 28)

        card_h_layout.addWidget(card)
        card_h_layout.addStretch(1)
        
        layout.addLayout(card_h_layout)
        layout.addStretch(1)

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microalgae Counter")
        self.resize(1100, 800)
        self.setMinimumSize(950, 750)
        self.setStyleSheet(STYLE + DIALOG_STYLE)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.main_scr = MainScreen(
            self.goto,
            self.run_process,
            self.browse_calibration_folder,  # For calibration folder
            self.browse_samples_folder          # For samples folder (reusing same method)
        )
        self.sett_scr = SettingsScreen(self.goto)
        self.def_scr = DefaultScreen(self.goto)

        self.stack.addWidget(self.main_scr)
        self.stack.addWidget(self.sett_scr)
        self.stack.addWidget(self.def_scr)
        
        self.processing_thread = None

    def goto(self, index):
        self.stack.setCurrentIndex(index)

    def browse_calibration_folder(self, line_edit):
        """Browse for calibration folder (with grid images)"""
        folder = QFileDialog.getExistingDirectory(self, "Выбрать папку с калибровочными изображениями")
        if folder:
            line_edit.setText(folder)
            # Validate it has images
            images = [f for f in os.listdir(folder)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
            if len(images) < 5:
                QMessageBox.warning(self, "Предупреждение",
                                    f"В папке только {len(images)} изображений.\n"
                                    "Для точной калибровки нужно минимум 5 фото сетки.")

    def browse_samples_folder(self, line_edit):
        """Browse for folder with cell images to analyze"""
        folder = QFileDialog.getExistingDirectory(self, "Выбрать папку с фото клеток для анализа")
        if folder:
            line_edit.setText(folder)


    def run_process(self):
        calib_folder = self.main_scr.path_calib.text().strip()
        samples_folder = self.main_scr.path_samples.text().strip()

        if not calib_folder or not samples_folder:
            QMessageBox.warning(self, "Внимание",
                                "Пожалуйста, выберите папку с калибровкой и папку с образцами.")
            return

        if not os.path.exists(calib_folder) or not os.path.exists(samples_folder):
            QMessageBox.critical(self, "Ошибка", "Указанные папки не существуют.")
            return

        # Count calibration images
        calib_images = [f for f in os.listdir(calib_folder)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
        if len(calib_images) < 5:
            reply = QMessageBox.question(self, "Мало калибровочных изображений",
                                         f"Найдено только {len(calib_images)} калибровочных изображений.\n"
                                         "Результат может быть неточным. Продолжить?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return

        # Count sample images
        sample_images = [f for f in os.listdir(samples_folder)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
                         and f != "segmented"]  # Exclude output folder
        if not sample_images:
            QMessageBox.warning(self, "Нет изображений",
                                "В папке с образцами нет изображений для анализа.")
            return

        # Show summary
        msg = (f"Найдено:\n"
               f"• Калибровочных изображений: {len(calib_images)}\n"
               f"• Образцов для анализа: {len(sample_images)}\n\n"
               f"Результаты будут сохранены в:\n{samples_folder}/segmented/")

        reply = QMessageBox.question(self, "Запуск анализа", msg + "\n\nНачать обработку?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No:
            return

        # Get grid size and depth from settings
        grid_size = float(self.sett_scr.inputs['размер сетки (мм)'].text())
        depth = float(self.sett_scr.inputs['глубина камеры (мм)'].text())

        # Process all images in the samples folder (like inference_example.py)
        self.process_samples_folder(samples_folder, calib_folder, grid_size, depth)

    def process_samples_folder(self, samples_folder, calib_folder, grid_size, depth):
        """Process all images in the samples folder (like inference_example.py)"""

        print("=" * 50)
        print("НАЧАЛО ОБРАБОТКИ")
        print(f"Папка образцов: {samples_folder}")
        print(f"Папка калибровки: {calib_folder}")
        print(f"Размер сетки: {grid_size}")
        print(f"Глубина: {depth}")
        print("=" * 50)

        # Get parameters from settings
        params = {}
        print("\nПАРАМЕТРЫ ИЗ НАСТРОЕК:")
        for name, widget in self.sett_scr.inputs.items():
            if isinstance(widget, QComboBox):
                params[name] = widget.currentText()
                print(f"  {name}: {params[name]} (QComboBox)")
            else:
                params[name] = widget.text().strip()
                print(f"  {name}: '{params[name]}' (QLineEdit)")

                # Проверяем числовые параметры на пустые значения
                if name in ['коэф. сглаживания', 'мин. расстояние (пикс.)', 'мин. радиус (пикс.)',
                            'макс. радиус (пикс.)', 'верхний порог', 'нижний порог']:
                    if not params[name]:
                        print(f"  ⚠️ ВНИМАНИЕ: {name} пустой! Использую значение по умолчанию")
                        params[name] = DEFAULT_VALUES.get(name, '')
                    try:
                        int(params[name])
                        print(f"  ✅ {name} = {params[name]} (корректное целое число)")
                    except ValueError:
                        print(f"  ❌ ОШИБКА: {name} = '{params[name]}' не является целым числом!")

        # Create output folder
        output_folder = os.path.join(samples_folder, "segmented")
        os.makedirs(output_folder, exist_ok=True)
        print(f"\nПапка для результатов: {output_folder}")

        # Prepare results dataframe
        results = []

        # Get list of images
        image_files = [f for f in os.listdir(samples_folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
                       and f != "segmented"]

        print(f"\nНайдено изображений для обработки: {len(image_files)}")
        for img in image_files:
            print(f"  - {img}")

        if not image_files:
            QMessageBox.warning(self, "Нет изображений", "В папке нет изображений для обработки")
            return

        # Setup progress
        total_images = len(image_files)
        self.progress = QProgressDialog(f"Обработка 0/{total_images} изображений...",
                                        "Отмена", 0, total_images, self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.show()

        # First, calculate volume using calibration folder
        self.progress.setLabelText("Калибровка объема...")
        print("\n" + "=" * 50)
        print("КАЛИБРОВКА ОБЪЕМА")
        print("=" * 50)

        # Get first image to get dimensions
        first_img_path = os.path.join(samples_folder, image_files[0])
        print(f"Загружаю первое изображение для определения размеров: {first_img_path}")
        first_img = cv2.imread(first_img_path)

        if first_img is None:
            error_msg = f"Не удалось загрузить {first_img_path}"
            print(f"❌ {error_msg}")
            QMessageBox.critical(self, "Ошибка", error_msg)
            return

        P_h, P_w = first_img.shape[:2]
        print(f"Размеры изображения: {P_h} x {P_w} пикселей")

        # Calculate volume using calibration images
        calib_plot_path = os.path.join(output_folder, "calibration_stats.png")
        print(f"Вычисление объема по калибровочным изображениям из: {calib_folder}")
        print(f"График калибровки будет сохранен в: {calib_plot_path}")

        try:
            v_img = calc_volume(
                imgs_path=calib_folder,
                l=grid_size,
                h=depth,
                P_h=P_h,
                P_w=P_w,
                plot_stats_path=calib_plot_path
            )
            print(f"✅ Объем изображения: {v_img:.6f} мм³")
        except Exception as e:
            print(f"❌ Ошибка при калибровке: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Ошибка калибровки", f"Не удалось вычислить объем:\n{str(e)}")
            return

        # Calculate scale factor
        s_squared = v_img / (P_h * P_w * depth)
        mm_per_pixel = math.sqrt(s_squared)
        print(f"Масштаб: {mm_per_pixel:.6e} мм/пиксель")

        # Process each image
        print("\n" + "=" * 50)
        print("ОБРАБОТКА ИЗОБРАЖЕНИЙ")
        print("=" * 50)

        for idx, image_file in enumerate(image_files):
            if self.progress.wasCanceled():
                print("Обработка отменена пользователем")
                break

            self.progress.setLabelText(f"Обработка {idx + 1}/{total_images}: {image_file}")
            self.progress.setValue(idx)
            QApplication.processEvents()

            print(f"\n--- Обработка {idx + 1}/{total_images}: {image_file} ---")

            try:
                # Load image
                img_path = os.path.join(samples_folder, image_file)
                print(f"Загрузка: {img_path}")
                img = cv2.imread(img_path)

                if img is None:
                    print(f"❌ Не удалось загрузить {image_file}")
                    continue

                print(f"✅ Изображение загружено, размер: {img.shape}")

                # Configure detection parameters
                increase_channel = None
                contrast = params.get('контрастность', 'нет').lower()
                print(f"Контрастность: {contrast}")

                if contrast == 'да':
                    increase_channel = 1
                    print("  Будет применено улучшение контраста (канал 1 - зеленый)")

                # Проверяем все числовые параметры перед использованием
                try:
                    minDist = int(params.get('мин. расстояние (пикс.)', 10))
                    print(f"мин. расстояние: {minDist}")
                except ValueError:
                    print(f"⚠️ Ошибка в параметре 'мин. расстояние', использую значение по умолчанию 10")
                    minDist = 10

                try:
                    minRadius = int(params.get('мин. радиус (пикс.)', 3))
                    print(f"мин. радиус: {minRadius}")
                except ValueError:
                    print(f"⚠️ Ошибка в параметре 'мин. радиус', использую значение по умолчанию 3")
                    minRadius = 3

                try:
                    maxRadius = int(params.get('макс. радиус (пикс.)', 20))
                    print(f"макс. радиус: {maxRadius}")
                except ValueError:
                    print(f"⚠️ Ошибка в параметре 'макс. радиус', использую значение по умолчанию 20")
                    maxRadius = 20

                try:
                    param2 = int(params.get('верхний порог', 20))
                    print(f"верхний порог: {param2}")
                except ValueError:
                    print(f"⚠️ Ошибка в параметре 'верхний порог', использую значение по умолчанию 20")
                    param2 = 20

                try:
                    blur_kernel = int(params.get('коэф. сглаживания', 3))
                    print(f"коэф. сглаживания: {blur_kernel}")
                except ValueError:
                    print(f"⚠️ Ошибка в параметре 'коэф. сглаживания', использую значение по умолчанию 3")
                    blur_kernel = 3

                # Detect cells
                print("Запуск detect_cells...")
                circles = detect_cells(
                    img,
                    increase_channel=increase_channel,
                    minDist=minDist,
                    minRadius=minRadius,
                    maxRadius=maxRadius,
                    param2=param2,
                    blur_kernel=blur_kernel
                )

                print(f"Результат detect_cells: circles.shape = {circles.shape}")

                # Count cells
                if circles.shape[1] == 0:
                    cell_count = 0
                    print(f"⚠️ Клетки не обнаружены в {image_file}")
                else:
                    cell_count = circles.shape[1]
                    print(f"✅ Обнаружено клеток: {cell_count}")

                    # Save visualization
                    output_img_path = os.path.join(output_folder, f"seg_{image_file}")
                    show_radius = params.get('средний радиус', 'нет').lower() == 'да'
                    show_annotation = params.get('обозначение', 'нет').lower() == 'да'

                    print(f"Сохранение визуализации в: {output_img_path}")
                    print(f"  показывать радиус: {show_radius}, аннотации: {show_annotation}")

                    visualize_circles(img, circles, save_path=output_img_path,
                                      annotate=show_annotation, ext_title=show_radius)
                    print(f"✅ Визуализация сохранена")

                # Calculate concentration
                try:
                    dilution = float(params.get('коэф. разбавления', 1))
                    print(f"коэф. разбавления: {dilution}")
                except ValueError:
                    print(f"⚠️ Ошибка в параметре 'коэф. разбавления', использую значение по умолчанию 1")
                    dilution = 1.0

                concentration = calculate_concentration(cell_count, dilution, v_img)
                print(f"Концентрация: {concentration} клеток/мл")

                # Store results
                results.append({
                    'image': image_file,
                    'cells_num': cell_count,
                    'concentration': concentration,
                    'volume_mm3': v_img,
                    'scale_mm_per_px': mm_per_pixel
                })
                print(f"✅ Результаты сохранены")

            except Exception as e:
                print(f"❌ ОШИБКА при обработке {image_file}:")
                import traceback
                traceback.print_exc()
                continue

        self.progress.close()

        # Save results to CSV (like in inference_example.py)
        if results:
            df = pd.DataFrame(results)
            csv_path = os.path.join(output_folder, "cells.csv")
            df.to_csv(csv_path, index=False)

            # Also save a summary text file
            summary_path = os.path.join(output_folder, "summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"РЕЗУЛЬТАТЫ АНАЛИЗА\n")
                f.write(f"=" * 50 + "\n\n")
                f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Папка с образцами: {samples_folder}\n")
                f.write(f"Папка калибровки: {calib_folder}\n")
                f.write(f"Размер сетки: {grid_size} мм\n")
                f.write(f"Глубина камеры: {depth} мм\n")
                f.write(f"Объем изображения: {v_img:.6f} мм³\n")
                f.write(f"Масштаб: {mm_per_pixel:.6e} мм/пиксель\n\n")
                f.write(f"Обработано изображений: {len(results)}\n\n")
                f.write(f"Параметры обнаружения:\n")
                for key, value in params.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n" + "=" * 50 + "\n\n")

                total_cells = sum(r['cells_num'] for r in results)
                avg_concentration = sum(r['concentration'] for r in results) / len(results)

                f.write(f"ВСЕГО КЛЕТОК: {total_cells}\n")
                f.write(f"СРЕДНЯЯ КОНЦЕНТРАЦИЯ: {avg_concentration:.0f} клеток/мл\n\n")

                f.write("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:\n")
                for r in results:
                    f.write(f"{r['image']}: {r['cells_num']} клеток, {r['concentration']} клеток/мл\n")

            # Show completion message
            msg = (f"✅ Обработка завершена!\n\n"
                   f"Обработано изображений: {len(results)}\n"
                   f"Всего клеток: {total_cells}\n"
                   f"Средняя концентрация: {avg_concentration:.0f} клеток/мл\n\n"
                   f"Результаты сохранены в:\n{output_folder}")

            QMessageBox.information(self, "Успех", msg)

            # Optionally open the output folder
            reply = QMessageBox.question(self, "Открыть папку",
                                         "Открыть папку с результатами?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                os.startfile(output_folder)  # Windows
                # For cross-platform: import subprocess; subprocess.Popen(f'explorer "{output_folder}"')
        else:
            QMessageBox.warning(self, "Нет результатов", "Не удалось обработать ни одного изображения.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec())
    
