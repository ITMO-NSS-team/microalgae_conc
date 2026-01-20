import sys
import os
from pathlib import Path
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                               QStackedWidget, QFrame, QGridLayout, QMessageBox,
                               QProgressDialog, QFileDialog)
from PySide6.QtCore import Qt, QThread, Signal
import cv2
import numpy as np

# ИМПОРТЫ ИЗ РЕПОЗИТОРИЯ
try:
    from CellCounter.Segmentator import detect_cells, visualize_circles
    from CellCounter.ConcentrationCalculator import calculate_concentration
    from CellCounter.VolumeCalculator import calc_volume
except ImportError:
    # Заглушка для теста интерфейса без библиотек
    print("ВНИМАНИЕ: Модули CellCounter не найдены. Функционал будет ограничен.")
    def detect_cells(*args, **kwargs): return np.zeros((1, 9, 3)) # имитация 9 клеток
    def visualize_circles(*args, **kwargs): pass
    def calculate_concentration(*args, **kwargs): return 150193
    def calc_volume(*args, **kwargs): return 0.011985

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
    "мин. радиус (мм)": "15",
    "макс. радиус (мм)": "100",
    "верхний порог": "30",
    "нижний порог": "30",
    "контрастность": "нет",
    "обозначение": "нет",
    "средний радиус": "нет"
}

MANDATORY_FIELDS = ["размер сетки (мм)", "глубина камеры (мм)", "коэф. разбавления"]

STYLE = f"""
    QMainWindow {{ 
        background-color: {COLOR_BG_MAIN}; 
    }}
    QWidget {{ 
        color: {COLOR_TEXT}; 
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif; 
        font-size: 16px; 
    }}
    
    QLineEdit {{
        background-color: {COLOR_INPUT_BG};
        border-radius: 8px;
        padding: 8px 12px;
        color: white;
        border: 1px solid rgba(255,255,255,0.3);
        font-size: 15px;
    }}
    QLineEdit:focus {{
        border: 1px solid white;
        background-color: {COLOR_ACCENT};
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
                minRadius=int(self.params.get('мин. радиус (мм)', 15)),
                maxRadius=int(self.params.get('макс. радиус (мм)', 100)),
                param2=int(self.params.get('верхний порог', 30)),
                blur_kernel=int(self.params.get('коэф. сглаживания', 25))
            )
            
            if circles.shape[1] == 0:
                raise ValueError("Клетки не обнаружены. Попробуйте изменить параметры порогов или радиуса.")
            
            cell_count = circles.shape[1]
            self.progress.emit(f"Обнаружено клеток: {cell_count}")
            
            self.progress.emit("Расчет объема...")
            grid_size = float(self.params['размер сетки (мм)'])
            depth = float(self.params['глубина камеры (мм)'])
            P_h, P_w = image.shape[:2]
            
            v_img = calc_volume(
                imgs_path=self.calibration_folder,
                l=grid_size,
                h=depth,
                P_h=P_h,
                P_w=P_w,
                plot_stats_path=None
            )
            
            self.progress.emit("Расчет концентрации...")
            dilution = float(self.params['коэф. разбавления'])
            concentration = calculate_concentration(cell_count, dilution, v_img)
            
            self.progress.emit("Сохранение результатов...")
            output_dir = Path(self.image_path).parent / "results"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"result_{Path(self.image_path).stem}.png"
            txt_output_path = output_dir / f"result_{Path(self.image_path).stem}.txt"
            
            show_radius = self.params.get('средний радиус', 'нет').lower() in ['да', 'yes']
            show_annotation = self.params.get('обозначение', 'нет').lower() in ['да', 'yes']
            
            visualize_circles(image, circles, save_path=str(output_path),
                              annotate=show_annotation, ext_title=show_radius)
            
            # Сохранение отчета в новом формате
            self.save_results_to_txt(txt_output_path, cell_count, v_img, concentration,
                                     self.image_path, self.calibration_folder, self.params)
            
            result = {
                'cell_count': cell_count,
                'concentration': concentration,
                'volume': v_img,
                'output_path': str(output_path),
                'txt_path': str(txt_output_path),
                'output_dir': str(output_dir) # путь к папке
            }
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def save_results_to_txt(self, filepath, cell_count, volume, concentration, 
                           image_path, calib_folder, params):
        """Сохранение отчета по указанному формату"""
        with open(filepath, 'w', encoding='utf-8') as f:
            # Заголовок с датой
            f.write(f"Дата и время анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Входные данные
            f.write("ВХОДНЫЕ ДАННЫЕ\n")
            f.write(f"Изображение для анализа: {image_path}\n")
            f.write(f"Папка калибровки: {calib_folder}\n\n")
            
            # Параметры обработки
            f.write("ПАРАМЕТРЫ ОБРАБОТКИ\n")
            f.write(f"Размер сетки: {params.get('размер сетки (мм)', '?')} мм\n")
            f.write(f"Глубина камеры: {params.get('глубина камеры (мм)', '?')} мм\n")
            f.write(f"Коэффициент разбавления: {params.get('коэф. разбавления', '?')}\n")
            f.write(f"Коэффициент сглаживания: {params.get('коэф. сглаживания', '25')}\n")
            f.write(f"Минимальный радиус клетки: {params.get('мин. радиус (мм)', '15')} пикс.\n")
            f.write(f"Максимальный радиус клетки: {params.get('макс. радиус (мм)', '100')} пикс.\n")
            f.write(f"Верхний порог: {params.get('верхний порог', '30')}\n")
            f.write(f"Повышение контрастности: {params.get('контрастность', 'нет')}\n\n")
            
            # Результаты
            f.write("РЕЗУЛЬТАТЫ АНАЛИЗА\n")
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

        for name in MANDATORY_FIELDS:
            row_w = QWidget()
            row = QHBoxLayout(row_w)
            row.setContentsMargins(0,0,0,0)
            
            lbl = QLabel(name)
            lbl.setWordWrap(True)
            
            edit = QLineEdit()
            edit.setFixedWidth(140)
            edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            self.inputs[name] = edit
            row.addWidget(lbl, 1)
            row.addWidget(edit, 0)
            left_layout.addWidget(row_w)
        
        left_layout.addStretch()
        
        btn_def = QPushButton('Посмотреть "по умолчанию" ➜')
        btn_def.setStyleSheet(f"background-color: {COLOR_BG_MAIN};")
        btn_def.clicked.connect(lambda: nav(2))
        left_layout.addWidget(btn_def)

        right_card = QFrame()
        right_card.setObjectName("Card")
        right_layout = QVBoxLayout(right_card)
        right_layout.setSpacing(12)

        r_title = QLabel("Дополнительные настройки")
        r_title.setObjectName("SectionTitle")
        r_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(r_title)
        right_layout.addWidget(QLabel("(можно изменить при необходимости)"), alignment=Qt.AlignmentFlag.AlignCenter)
        right_layout.addSpacing(10)
        
        for name in DEFAULT_VALUES.keys():
            row_w = QWidget()
            row = QHBoxLayout(row_w)
            row.setContentsMargins(0,0,0,0)
            
            lbl = QLabel(name)
            lbl.setWordWrap(True)
            
            edit = QLineEdit()
            edit.setFixedWidth(140)
            edit.setPlaceholderText(str(DEFAULT_VALUES[name]))
            edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
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
        layout.setContentsMargins(40, 40, 40, 40)
        
        header = QHBoxLayout()
        btn_back = QPushButton("← Назад к настройкам")
        btn_back.setFixedWidth(200)
        btn_back.clicked.connect(lambda: nav(1))
        header.addWidget(btn_back)
        header.addStretch()
        layout.addLayout(header)

        layout.addStretch(1)

        card = QFrame()
        card.setObjectName("Card")
        card.setMinimumWidth(500)
        card.setMinimumHeight(400)
        
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(40)
        grid.setVerticalSpacing(20)
        grid.setContentsMargins(40, 40, 40, 40)

        title = QLabel("Параметры по умолчанию")
        title.setObjectName("SectionTitle")
        title.setStyleSheet("font-size: 24px; margin-bottom: 20px;")
        grid.addWidget(title, 0, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)

        row = 1
        for k, v in DEFAULT_VALUES.items():
            k_label = QLabel(k)
            k_label.setStyleSheet("color: #ddd; font-weight: bold;")
            
            v_label = QLabel(v)
            v_label.setStyleSheet("color: #fff; font-size: 18px;")
            v_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            
            grid.addWidget(k_label, row, 0)
            grid.addWidget(v_label, row, 1)
            
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setStyleSheet("color: rgba(255,255,255,0.1);")
            grid.addWidget(line, row+1, 0, 1, 2)
            
            row += 2

        info_l = QLabel("Канал контраста")
        info_l.setStyleSheet("color: #ddd; font-weight: bold;")
        info_v = QLabel("0 (Синий)")
        info_v.setStyleSheet("color: #fff; font-size: 18px;")
        info_v.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        grid.addWidget(info_l, row, 0)
        grid.addWidget(info_v, row, 1)

        layout.addWidget(card, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microalgae Concentration Calculator")
        self.resize(1100, 800)
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
        
        for name, edit in self.sett_scr.inputs.items():
            val = edit.text().strip()
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
                final_params[name] = val if val else DEFAULT_VALUES[name]

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
        # Вывод сообщения с полным путем
        msg = (f"Анализ успешно завершен!\n\n"
               f"Обнаружено клеток: {res['cell_count']}\n"
               f"Концентрация: {res['concentration']} клеток/мл\n\n"
               f"Файлы сохранены в папку:\n{res['output_dir']}")
        QMessageBox.information(self, "Успех", msg)
        
    def on_error(self, err):
        self.progress.close()
        QMessageBox.critical(self, "Ошибка обработки", err)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec())
    