import tkinter as tk
from tkinter import messagebox, filedialog
from gtts import gTTS
import os
from playsound import playsound
from pydub import AudioSegment


def text_to_audio(speed, volume):
    text = text_entry.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Attention", "Veuillez entrer un texte à convertir.")
        return
    
    try:
        tts = gTTS(text=text, lang='fr', slow=speed)  
        tts.save("output.mp3")

        # Charger le fichier audio généré
        audio = AudioSegment.from_mp3("output.mp3")

        # Ajuster le volume
        audio = audio + volume  # Ajustez le volume ici

        # Enregistrer le fichier audio avec le volume ajusté
        audio.export("output_adjusted.mp3", format="mp3")

        messagebox.showinfo("Succès", "Texte converti en audio avec succès.")
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur est survenue : {e}")


# Fonction pour lire l'audio généré
def play_audio():
    audio_file = "output_adjusted.mp3"
    if os.path.exists(audio_file):
        try:
            playsound(os.path.abspath(audio_file))
        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur est survenue lors de la lecture de l'audio : {e}")
    else:
        messagebox.showwarning("Attention", "Le fichier audio n'existe pas. Veuillez d'abord convertir le texte.")


# Fonction pour sélectionner un fichier texte à charger dans le widget Text
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Fichiers texte", "*.txt")])
    if file_path:
        with open(file_path, "r") as file:
            text = file.read()
        text_entry.delete("1.0", tk.END)
        text_entry.insert(tk.END, text)


# Fonction pour enregistrer le texte converti en favoris
def save_as_favorite():
    text = text_entry.get("1.0", tk.END).strip()
    if text:
        with open("favorites.txt", "a") as file:
            file.write(text + "\n")
        messagebox.showinfo("Succès", "Texte sauvegardé en tant que favori.")


# Fonction pour charger les favoris
def load_favorites():
    if os.path.exists("favorites.txt"):
        with open("favorites.txt", "r") as file:
            favorites = file.readlines()
        favorites_window = tk.Toplevel(root)
        favorites_window.title("Favoris")
        favorites_window.geometry("300x200")
        favorites_text = tk.Text(favorites_window)
        favorites_text.pack(expand=True, fill="both")
        for favorite in favorites:
            favorites_text.insert(tk.END, favorite)



# Initialisation des variables globales pour la vitesse et le volume
conversion_speed = False
audio_volume = 0

# Création de la fenêtre principale
root = tk.Tk()
root.title("Transcription Texte en Audio")
root.geometry("400x400")

# Création de la barre de menu
menubar = tk.Menu(root)
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Exit", command=root.destroy)
menubar.add_cascade(label="File", menu=filemenu)

aboutmenu = tk.Menu(menubar, tearoff=0)
aboutmenu.add_command(label="About", command=lambda: messagebox.showinfo("About", "Ce programme est une application de transcription de texte en audio, réalisée par Ait Salim Karim et Tebbaa El Yazid, deux étudiants en 1A GD à l'ENSIAS."))
aboutmenu.add_command(label="Acknowledgements", command=lambda: messagebox.showinfo("Acknowledgements", "On tient tout d’abord à remercier Mme Sanaa EL FKIHI pour son soutien constant, ses précieux conseils et son expertise qui ont grandement contribué à l’avancement de ce projet. Sa disponibilité et son engagement ont été d’une aide précieuse tout au long de cette réalisation. Nos remerciements vont également à l’ENSIAS pour avoir mis à notre disposition les ressources nécessaires à la réalisation de ce projet. Leur soutien logistique a été un atout majeur dans la concrétisation de nos idées."))
menubar.add_cascade(label="Others", menu=aboutmenu)

root.config(menu=menubar)

# Création du widget Text avec coloration syntaxique
text_label = tk.Label(root, text="Entrez votre texte :")
text_label.pack(pady=10)
text_entry = tk.Text(root, height=10, width=40, wrap="word", font=("Courier", 10))
text_entry.pack(pady=10)

# Ajout des boutons pour les actions principales
convert_button = tk.Button(root, text="Convertir en Audio", command=lambda: text_to_audio(conversion_speed, audio_volume), bg="lightblue", fg="black")
convert_button.pack(pady=5)

play_button = tk.Button(root, text="Lire l'Audio", command=play_audio, bg="lightgreen", fg="black")
play_button.pack(pady=5)

file_button = tk.Button(root, text="Sélectionner un Fichier", command=select_file)
file_button.pack(pady=5)

# Ajout des boutons pour les fonctionnalités supplémentaires
favorites_button = tk.Button(root, text="Sauvegarder comme Favori", command=save_as_favorite, bg="lightyellow", fg="black")
favorites_button.pack(pady=5)
load_favorites_button = tk.Button(root, text="Charger les Favoris", command=load_favorites, bg="lightyellow", fg="black")
load_favorites_button.pack(pady=5)


# Lancement de l'interface graphique
root.mainloop()
