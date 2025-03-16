import cv2
from mtcnn import MTCNN

# Initialiser la webcam
cap = cv2.VideoCapture(0)  # 0 pour la webcam intégrée, 1 si tu as une webcam externe

# Initialiser le détecteur MTCNN
detector = MTCNN()

while True:
    # Lire une image depuis la webcam
    ret, frame = cap.read()
    if not ret:
        break  # Sort si la webcam ne fonctionne pas

    # Détecter les visages
    faces = detector.detect_faces(frame)

    # Dessiner des rectangles autour des visages détectés
    for face in faces:
        x, y, width, height = face["box"]
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)

    # Afficher l'image avec les visages détectés
    cv2.imshow("Détection de Visage en Temps Réel", frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la webcam et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()