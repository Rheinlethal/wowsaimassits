import math
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
import csv

Window.size = (800, 600)

class WOWSAimCalculator(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 10
        self.spacing = 10
        
        # Load ship data
        self.ships = self.load_ships()
        self.ship_names = sorted([ship['ship_name'] for ship in self.ships])
        
        # Title
        title = Label(
            text='WOWS Aim Calculator',
            size_hint_y=None,
            height=40,
            font_size='20sp',
            bold=True
        )
        self.add_widget(title)
        
        # Main content in scrollview
        scroll = ScrollView(size_hint=(1, 1))
        content = BoxLayout(orientation='vertical', spacing=10, size_hint_y=None)
        content.bind(minimum_height=content.setter('height'))
        
        # Ship selection section
        ship_layout = GridLayout(cols=2, spacing=10, size_hint_y=None, height=100)
        
        ship_layout.add_widget(Label(text='Kapal Anda:', size_hint_x=0.3))
        self.my_ship = TextInput(
            hint_text='Ketik nama kapal Anda...',
            multiline=False,
            size_hint_x=0.7
        )
        self.my_ship.bind(text=self.on_my_ship_text)
        ship_layout.add_widget(self.my_ship)
        
        ship_layout.add_widget(Label(text='Kapal Musuh:', size_hint_x=0.3))
        self.enemy_ship = TextInput(
            hint_text='Ketik nama kapal musuh...',
            multiline=False,
            size_hint_x=0.7
        )
        self.enemy_ship.bind(text=self.on_enemy_ship_text)
        ship_layout.add_widget(self.enemy_ship)
        
        content.add_widget(ship_layout)
        
        # Suggestion buttons for ships
        self.my_ship_suggestions_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=40,
            spacing=5
        )
        content.add_widget(self.my_ship_suggestions_layout)
        
        self.enemy_ship_suggestions_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=40,
            spacing=5
        )
        content.add_widget(self.enemy_ship_suggestions_layout)
        
        # Input section
        input_layout = GridLayout(cols=2, spacing=10, size_hint_y=None, height=200)
        
        input_layout.add_widget(Label(text='Jarak ke Target (km):', size_hint_x=0.4))
        self.distance_input = TextInput(
            hint_text='Contoh: 15.5',
            multiline=False,
            input_filter='float',
            size_hint_x=0.6
        )
        input_layout.add_widget(self.distance_input)
        
        input_layout.add_widget(Label(text='Arah Musuh (0-180°):', size_hint_x=0.4))
        self.angle_input = TextInput(
            hint_text='0° = menghadap, 90° = tegak lurus, 180° = membelakangi',
            multiline=False,
            input_filter='float',
            size_hint_x=0.6
        )
        input_layout.add_widget(self.angle_input)
        
        input_layout.add_widget(Label(text='Tipe Peluru:', size_hint_x=0.4))
        self.ammo_type = Spinner(
            text='AP',
            values=['AP', 'HE'],
            size_hint_x=0.6
        )
        input_layout.add_widget(self.ammo_type)
        
        content.add_widget(input_layout)
        
        # Calculate button
        calc_btn = Button(
            text='HITUNG AIM',
            size_hint_y=None,
            height=50,
            background_color=(0.2, 0.6, 0.8, 1),
            bold=True
        )
        calc_btn.bind(on_press=self.calculate_aim)
        content.add_widget(calc_btn)
        
        # Result section
        self.result_label = Label(
            text='',
            size_hint_y=None,
            height=300,
            markup=True,
            valign='top',
            halign='left'
        )
        self.result_label.bind(size=self._update_text_size)
        content.add_widget(self.result_label)
        
        scroll.add_widget(content)
        self.add_widget(scroll)
    
    def _update_text_size(self, instance, value):
        instance.text_size = (instance.width - 20, None)
    
    def on_my_ship_text(self, instance, value):
        """Update suggestions when typing ship name"""
        self.my_ship_suggestions_layout.clear_widgets()
        
        if len(value) >= 2:
            matches = [name for name in self.ship_names if value.lower() in name.lower()]
            if matches:
                # Show first 5 matches as buttons
                for ship_name in matches[:5]:
                    btn = Button(
                        text=ship_name,
                        size_hint_x=None,
                        width=150,
                        background_color=(0.2, 0.5, 0.8, 1),
                        font_size='11sp'
                    )
                    btn.bind(on_press=lambda x, name=ship_name: self.select_my_ship(name))
                    self.my_ship_suggestions_layout.add_widget(btn)
                
                # Show count if more matches
                if len(matches) > 5:
                    count_label = Label(
                        text=f'+{len(matches)-5} lagi',
                        size_hint_x=None,
                        width=80,
                        color=(0.7, 0.7, 0.7, 1),
                        font_size='11sp'
                    )
                    self.my_ship_suggestions_layout.add_widget(count_label)
            else:
                no_match = Label(
                    text='Tidak ada kapal yang cocok',
                    color=(1, 0.3, 0.3, 1),
                    font_size='11sp'
                )
                self.my_ship_suggestions_layout.add_widget(no_match)
    
    def on_enemy_ship_text(self, instance, value):
        """Update suggestions when typing enemy ship name"""
        self.enemy_ship_suggestions_layout.clear_widgets()
        
        if len(value) >= 2:
            matches = [name for name in self.ship_names if value.lower() in name.lower()]
            if matches:
                # Show first 5 matches as buttons
                for ship_name in matches[:5]:
                    btn = Button(
                        text=ship_name,
                        size_hint_x=None,
                        width=150,
                        background_color=(0.8, 0.5, 0.2, 1),
                        font_size='11sp'
                    )
                    btn.bind(on_press=lambda x, name=ship_name: self.select_enemy_ship(name))
                    self.enemy_ship_suggestions_layout.add_widget(btn)
                
                # Show count if more matches
                if len(matches) > 5:
                    count_label = Label(
                        text=f'+{len(matches)-5} lagi',
                        size_hint_x=None,
                        width=80,
                        color=(0.7, 0.7, 0.7, 1),
                        font_size='11sp'
                    )
                    self.enemy_ship_suggestions_layout.add_widget(count_label)
            else:
                no_match = Label(
                    text='Tidak ada kapal yang cocok',
                    color=(1, 0.3, 0.3, 1),
                    font_size='11sp'
                )
                self.enemy_ship_suggestions_layout.add_widget(no_match)
    
    def select_my_ship(self, ship_name):
        """Select ship for player"""
        self.my_ship.text = ship_name
        self.my_ship_suggestions_layout.clear_widgets()
    
    def select_enemy_ship(self, ship_name):
        """Select enemy ship"""
        self.enemy_ship.text = ship_name
        self.enemy_ship_suggestions_layout.clear_widgets()
    
    def load_ships(self):
        """Load ship data from CSV"""
        ships = []
        try:
            with open('ship_datasets.csv', 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # Skip rows with empty critical fields
                        if not row['ship_name'] or not row['initial_he_velocity'] or not row['initial_ap_velocity']:
                            continue
                        
                        # Convert to float with default value if empty
                        def safe_float(value, default=0.0):
                            try:
                                return float(value) if value and value.strip() else default
                            except ValueError:
                                return default
                        
                        ships.append({
                            'ship_name': row['ship_name'].strip(),
                            'ship_id': row['ship_id'].strip(),
                            'country': row['country'].strip(),
                            'maximum_dispersion': safe_float(row['maximum_dispersion']),
                            'firing_range': safe_float(row['firing_range']),
                            'initial_he_velocity': safe_float(row['initial_he_velocity']),
                            'initial_ap_velocity': safe_float(row['initial_ap_velocity']),
                            'maximum_speed': safe_float(row['maximum_speed'], 30.0)
                        })
                    except Exception as e:
                        print(f"Error loading ship {row.get('ship_name', 'Unknown')}: {e}")
                        continue
        except FileNotFoundError:
            print("File ship_datasets.csv tidak ditemukan!")
        except Exception as e:
            print(f"Error reading CSV: {e}")
        return ships
    
    def get_ship_data(self, ship_name):
        """Get ship data by name (case insensitive, partial match)"""
        # First try exact match
        for ship in self.ships:
            if ship['ship_name'].lower() == ship_name.lower():
                return ship
        
        # Then try partial match
        for ship in self.ships:
            if ship_name.lower() in ship['ship_name'].lower():
                return ship
        
        return None
    
    def calculate_aim(self, instance):
        """Calculate aim angle based on inputs"""
        try:
            # Validate inputs
            if not self.my_ship.text or self.my_ship.text.strip() == '':
                self.result_label.text = '[color=ff0000]Error: Masukkan nama kapal Anda![/color]'
                return
            
            if not self.enemy_ship.text or self.enemy_ship.text.strip() == '':
                self.result_label.text = '[color=ff0000]Error: Masukkan nama kapal musuh![/color]'
                return
            
            distance = float(self.distance_input.text)
            enemy_angle = float(self.angle_input.text)
            
            if not (0 <= enemy_angle <= 180):
                self.result_label.text = '[color=ff0000]Error: Arah musuh harus antara 0-180°![/color]'
                return
            
            # Get ship data
            my_ship_data = self.get_ship_data(self.my_ship.text.strip())
            enemy_ship_data = self.get_ship_data(self.enemy_ship.text.strip())
            
            if not my_ship_data:
                self.result_label.text = f'[color=ff0000]Error: Kapal "{self.my_ship.text}" tidak ditemukan!\nCek ejaan atau lihat saran di atas.[/color]'
                return
            
            if not enemy_ship_data:
                self.result_label.text = f'[color=ff0000]Error: Kapal "{self.enemy_ship.text}" tidak ditemukan!\nCek ejaan atau lihat saran di atas.[/color]'
                return
            
            # Get shell velocity based on ammo type
            if self.ammo_type.text == 'AP':
                shell_velocity = my_ship_data['initial_ap_velocity']
            else:
                shell_velocity = my_ship_data['initial_he_velocity']
            
            # Calculate travel time (simplified, assuming flat trajectory)
            distance_m = distance * 1000  # Convert km to meters
            travel_time = distance_m / shell_velocity
            
            # Enemy max speed in m/s
            enemy_speed_knots = enemy_ship_data['maximum_speed']
            enemy_speed_ms = enemy_speed_knots * 0.514444  # Convert knots to m/s
            
            # Calculate for different speed settings
            speed_settings = {
                '1/4 speed': 0.25,
                '1/2 speed': 0.50,
                '3/4 speed': 0.75,
                'Full speed': 1.00
            }
            
            results = []
            results.append('[b][size=18]HASIL KALKULASI AIM[/size][/b]\n')
            results.append(f'[b]Kapal Anda:[/b] {self.my_ship.text}')
            results.append(f'[b]Kapal Musuh:[/b] {self.enemy_ship.text}')
            results.append(f'[b]Jarak:[/b] {distance} km')
            results.append(f'[b]Arah Musuh:[/b] {enemy_angle}°')
            results.append(f'[b]Peluru:[/b] {self.ammo_type.text}')
            results.append(f'[b]Kecepatan Peluru:[/b] {shell_velocity:.0f} m/s')
            results.append(f'[b]Waktu Tempuh:[/b] {travel_time:.2f} detik\n')
            
            results.append('[b][size=16]LEAD ANGLE (Derajat Bidikan):[/size][/b]\n')
            
            for speed_name, speed_mult in speed_settings.items():
                # Calculate enemy displacement during shell travel
                enemy_speed = enemy_speed_ms * speed_mult
                displacement = enemy_speed * travel_time
                
                # Calculate the component of movement perpendicular to line of fire
                # enemy_angle: 0° = approaching, 90° = broadside, 180° = fleeing
                perpendicular_component = displacement * math.sin(math.radians(enemy_angle))
                
                # Calculate lead angle
                if distance_m > 0:
                    lead_angle = math.degrees(math.atan(perpendicular_component / distance_m))
                else:
                    lead_angle = 0
                
                # Determine direction
                if enemy_angle < 90:
                    direction = 'KIRI' if lead_angle > 0 else 'KANAN'
                else:
                    direction = 'KANAN' if lead_angle > 0 else 'KIRI'
                
                lead_angle_abs = abs(lead_angle)
                
                results.append(f'[b]{speed_name}:[/b] {lead_angle_abs:.2f}° ke {direction}')
            
            results.append('\n[color=00ff00][i]Tips: Arahkan aim sesuai derajat dan arah yang ditampilkan[/i][/color]')
            
            self.result_label.text = '\n'.join(results)
            
        except ValueError as e:
            self.result_label.text = f'[color=ff0000]Error: Input tidak valid! Pastikan semua field terisi dengan benar.[/color]'
        except Exception as e:
            self.result_label.text = f'[color=ff0000]Error: {str(e)}[/color]'

class WOWSAimApp(App):
    def build(self):
        return WOWSAimCalculator()

if __name__ == '__main__':
    WOWSAimApp().run()