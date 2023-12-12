import cv2
import numpy as np

# Functie om de kleur van het snelheidsbord te herkennen
def detect_speed_limit_color(image):
    # Converteer de afbeelding naar de HSV-kleurruimte
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definieer kleurbereik voor rode kleuren
    lower_red = (0, 100, 25)
    upper_red = (10, 255, 255)
    lower_red2 = (170, 100, 25)
    upper_red2 = (180, 255, 255)

    # Creëer maskers voor de rode kleur
    mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    # Combineer de maskers om alle rode tinten op te vangen
    combined_mask = cv2.bitwise_or(mask_red, mask_red2)

    return combined_mask

def find_circular_contours(contours, hierarchy):
    circular_contours = []
    for contour, h in zip(contours, hierarchy[0]):
        # Negeer innerlijke contouren (h[3] != -1 betekent dat het contour een kind heeft)
        if h[3] != -1:
            continue
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if 0.6 < circularity < 1.4:
            circular_contours.append(contour)
    return circular_contours


def detect_traffic_sign_edges_based_on_color(image):
    # Detecteer de kleur van het snelheidsbord
    speed_limit_mask = detect_speed_limit_color(image)

    # Voer randdetectie uit op basis van het kleurmasker
    edges = cv2.Canny(speed_limit_mask, 100, 200)

    return edges

def detect_traffic_sign_shape_based_on_color(image):
    # Detecteer de kleur van het snelheidsbord
    speed_limit_mask = detect_speed_limit_color(image)

    # Vind contouren in het masker
    contours, hierarchy = cv2.findContours(speed_limit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contouren die op een cirkel lijken
    circular_contours = find_circular_contours(contours, hierarchy)

    # Creëer een leeg beeld om de contouren te tekenen
    #contour_img = np.zeros_like(image)

    # Teken de gevonden cirkelvormige contouren
    cv2.drawContours(image, circular_contours, -1, (0, 255, 0), 2)

    return image

def crop_to_speed_limit_sign(image, circular_contours):
    # We gaan uit van één bord in de afbeelding voor eenvoud.
    # Zoek de bounding box van de grootste contour, aangenomen dat dit het bord is.
    if not circular_contours:
        return None  # Geen borden gevonden

    largest_contour = max(circular_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop de afbeelding op de bounding box van het snelheidsbord
    cropped_image = image[y:y+h, x:x+w]

    return cropped_image

# Voorbeeldgebruik
if __name__ == "__main__":
    image = cv2.imread("test.png")
    speed_limit_mask = detect_speed_limit_color(image)
    contours, hierarchy = cv2.findContours(speed_limit_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    circular_contours = find_circular_contours(contours, hierarchy)
    draw_contour = detect_traffic_sign_shape_based_on_color(image)
    cropped_sign = crop_to_speed_limit_sign(image, circular_contours)
    color_mask = detect_traffic_sign_edges_based_on_color(draw_contour)
    cv2.imshow("Originele Afbeelding", draw_contour)
    cv2.imshow("color mask", color_mask)
    if cropped_sign is not None:
        cv2.imshow("Gecropte Snelheidsbord", cropped_sign)

        cv2.waitKey(0)

    cv2.destroyAllWindows()





