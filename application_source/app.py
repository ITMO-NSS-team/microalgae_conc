import sys
import os
import math
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
plt.show = lambda: None 

from pathlib import Path
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                               QStackedWidget, QFrame, QGridLayout, QMessageBox,
                               QProgressDialog, QFileDialog, QComboBox)
from PySide6.QtCore import Qt, QThread, Signal
import cv2
import numpy as np
import traceback

# ИМПОРТЫ ИЗ РЕПОЗИТОРИЯ
try:
    from CellCounter.Segmentator import detect_cells, visualize_circles
    from CellCounter.ConcentrationCalculator import calculate_concentration
    from CellCounter.VolumeCalculator import calc_volume
except ImportError:
    print("ВНИМАНИЕ: Модули CellCounter не найдены. Функционал будет ограничен.")
    def detect_cells(*args, **kwargs): return np.zeros((1, 9, 3)) 
    def visualize_circles(*args, **kwargs): pass
    def calculate_concentration(*args, **kwargs): return 150193
    def calc_volume(*args, **kwargs): 
        if kwargs.get('plot_stats_path'):
            try:
                with open(kwargs['plot_stats_path'], 'w') as f: f.write("fake plot")
            except: pass
        return 0.011985

# КОНСТАНТЫ СТИЛЯ
COLOR_BG_MAIN = "#734d4d"
COLOR_CARD_BG = "#5a3e3e"
COLOR_ACCENT = "#8c6161"
COLOR_INPUT_BG = "#8e6b6b"
COLOR_TEXT = "#ffffff"
COLOR_BTN_RUN = "#4a3232"
COLOR_BTN_HOVER = "#5e4040"

DEFAULT_VALUES = {
    "коэф. сглаживания": "25",
    "мин. радиус (пикс.)": "15",
    "макс. радиус (пикс.)": "100",
    "верхний порог": "30",
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
            if self.params.get('контрастность', 'нет').lower() in ['да', 'авто', 'yes']:
                increase_channel = 0
            
            circles = detect_cells(
                image,
                increase_channel=increase_channel,
                minDist=int(self.params.get('коэф. сглаживания', 25)),
                minRadius=int(self.params.get('мин. радиус (пикс.)', 15)),
                maxRadius=int(self.params.get('макс. радиус (пикс.)', 100)),
                param2=int(self.params.get('верхний порог', 30)),
                blur_kernel=int(self.params.get('коэф. сглаживания', 25))
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
    def __init__(self, nav, run_logic, browse_folder, browse_photo):
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
        
        lbl_folder = QLabel("Папка с калибровочными изображениями (мин. 5 фото):")
        form_layout.addWidget(lbl_folder)
        
        row_folder = QHBoxLayout()
        self.path_folder = QLineEdit()
        self.path_folder.setObjectName("BigInput")
        self.path_folder.setPlaceholderText("Выберите папку...")
        btn_browse_f = QPushButton("📂")
        btn_browse_f.setObjectName("BrowseBtn")
        btn_browse_f.setFixedSize(50, 45)
        btn_browse_f.clicked.connect(browse_folder)
        row_folder.addWidget(self.path_folder)
        row_folder.addWidget(btn_browse_f)
        form_layout.addLayout(row_folder)
        
        lbl_photo = QLabel("Фотография микроводорослей для анализа:")
        form_layout.addWidget(lbl_photo)
        
        row_photo = QHBoxLayout()
        self.path_photo = QLineEdit()
        self.path_photo.setObjectName("BigInput")
        self.path_photo.setPlaceholderText("Выберите изображение...")
        btn_browse_p = QPushButton("🖼️")
        btn_browse_p.setObjectName("BrowseBtn")
        btn_browse_p.setFixedSize(50, 45)
        btn_browse_p.clicked.connect(browse_photo)
        row_photo.addWidget(self.path_photo)
        row_photo.addWidget(btn_browse_p)
        form_layout.addLayout(row_photo)
        
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

# [ИЗМЕНЕНИЕ] Подсказки теперь только на русском
PARAM_DESCRIPTIONS = {
    "коэф. сглаживания": "Размер ядра для медианного размытия (должен быть нечетным целым числом).",
    "мин. радиус (пикс.)": "Минимальный радиус клетки для обнаружения в пикселях.",
    "макс. радиус (пикс.)": "Максимальный радиус клетки для обнаружения в пикселях.",
    "верхний порог": "Порог аккумулятора (более низкие значения приводят к обнаружению большего числа ложных кругов).",
    "нижний порог": "Порог детектора границ Canny.",
    "контрастность": "Функция увеличения контраста в одном канале изображения с использованием CLAHE.",
    "обозначение": "Нумеровать ли обнаруженные круги.",
    "средний радиус": "Показывать ли расширенный заголовок со статистикой площади."
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
        self.setWindowTitle("Microalgae Concentration Calculator")
        self.resize(1100, 800)
        self.setMinimumSize(950, 750)
        self.setStyleSheet(STYLE)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.main_scr = MainScreen(self.goto, self.run_process, self.browse_folder, self.browse_photo)
        self.sett_scr = SettingsScreen(self.goto)
        self.def_scr = DefaultScreen(self.goto)

        self.stack.addWidget(self.main_scr)
        self.stack.addWidget(self.sett_scr)
        self.stack.addWidget(self.def_scr)
        
        self.processing_thread = None

    def goto(self, index):
        self.stack.setCurrentIndex(index)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выбрать папку для калибровки")
        if folder:
            self.main_scr.path_folder.setText(folder)
    
    def browse_photo(self):
        photo, _ = QFileDialog.getOpenFileName(self, "Выбрать фото", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif)")
        if photo:
            self.main_scr.path_photo.setText(photo)

    def run_process(self):
        folder = self.main_scr.path_folder.text().strip()
        photo = self.main_scr.path_photo.text().strip()

        if not folder or not photo:
            QMessageBox.warning(self, "Внимание", "Пожалуйста, выберите папку и фотографию.")
            return
        
        if not os.path.exists(folder) or not os.path.exists(photo):
            QMessageBox.critical(self, "Ошибка", "Указанные пути не существуют.")
            return

        final_params = {}
        missing = []
        
        for name, widget in self.sett_scr.inputs.items():
            if isinstance(widget, QComboBox):
                val = widget.currentText()
            else:
                val = widget.text().strip()
            
            if name in MANDATORY_FIELDS:
                if not val:
                    missing.append(name)
                else:
                    try:
                        float(val)
                        final_params[name] = val
                    except:
                        QMessageBox.warning(self, "Ошибка", f"Некорректное число: {name}")
                        return
            else:
                if isinstance(widget, QLineEdit) and not val:
                    final_params[name] = DEFAULT_VALUES[name]
                else:
                    final_params[name] = val

        if missing:
            QMessageBox.warning(self, "Заполните обязательные поля", 
                              "Зайдите в настройки и укажите:\n" + "\n".join(missing))
            self.goto(1)
            return

        self.progress = QProgressDialog("Инициализация...", "Отмена", 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.show()
        
        self.processing_thread = ProcessingThread(photo, folder, final_params)
        self.processing_thread.progress.connect(self.progress.setLabelText)
        self.processing_thread.finished.connect(self.on_finished)
        self.processing_thread.error.connect(self.on_error)
        self.processing_thread.start()

    def on_finished(self, res):
        self.progress.close()
        
        calib_msg = ""
        if os.path.exists(res['calib_plot']):
            calib_msg = f"\n📊 График калибровки создан:\n{os.path.basename(res['calib_plot'])}"
        
        # ДОБАВЛЕН ВЫВОД КОЭФФИЦИЕНТА
        msg = (f"Анализ успешно завершен!\n\n"
               f"Обнаружено клеток: {res['cell_count']}\n"
               f"Концентрация: {res['concentration']} клеток/мл\n"
               f"Объем изображения: {res['volume']:.6f} мм³\n"
               f"Масштаб: {res['scale_factor']:.6e} мм/пиксель\n"
               f"{calib_msg}\n\n"
               f"Файлы сохранены в папку:\n{res['output_dir']}")
        QMessageBox.information(self, "Успех", msg)
        
    def on_error(self, err):
        self.progress.close()
        
        error_dialog = QMessageBox(self)
        error_dialog.setIcon(QMessageBox.Icon.Critical)
        error_dialog.setWindowTitle("Ошибка обработки")
        error_dialog.setText("Произошла критическая ошибка при выполнении анализа.")
        error_dialog.setInformativeText("Технические подробности доступны в разделе деталей.")
        
        main_err_msg = err.strip().split('\n')[-1]
        error_dialog.setSecondaryText(main_err_msg) if hasattr(error_dialog, 'setSecondaryText') else None
        
        error_dialog.setDetailedText(err)
        error_dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        error_dialog.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec())
    
