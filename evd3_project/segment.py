import cv2
import os
import glob
import numpy as np

# grab the list of images in our data directory
print("[INFO] loading images...")
p = os.path.sep.join(['*.[jp][np][ge]*'])

file_list = [f for f in glob.iglob(p, recursive=True) if (os.path.isfile(f))]
print("[INFO] images found: {}".format(len(file_list)))

for filename in file_list:

    # Laden van de afbeelding
    image = cv2.imread(filename)
    cv2.imshow("orginele image", image)
    # Converteer de afbeelding naar de HSV-kleurruimte
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definieer kleurbereik voor rode kleuren
    lower_red = (0, 100, 50)
    upper_red = (10, 255, 255)
    lower_red2 = (170, 100, 50)
    upper_red2 = (180, 255, 255)

    mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    # Combineer de maskers
    combined_mask = cv2.bitwise_or(mask_red, mask_red2)


    # Identificeer contouren in het masker
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) >= 8:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))

            # Controleer of de gedetecteerde cirkel overeenkomt met een snelheidsbord
            if radius >= 10:
                # Maak een masker voor het snelheidsbord
                mask_speed = image
                mask_speed_sign = np.zeros_like(image)

                cv2.circle(mask_speed_sign, center, int(radius), (255, 255, 255), thickness=cv2.FILLED)

                cv2.circle(mask_speed, center, int(radius), (0, 255, 0), thickness=2)

                cv2.imshow('contour', mask_speed)

                # Maskeer de oorspronkelijke afbeelding met het snelheidsbordmasker
                result = cv2.bitwise_and(image, mask_speed_sign)

                # Bepaal de begrenzende rechthoek van het snelheidsbord
                x, y, w, h = cv2.boundingRect(contour)
                # Snijd het snelheidsbordgedeelte uit de afbeelding
                cropped_speed_sign = result[y:y + h, x:x + w]


                # Toon het gesegmenteerde snelheidsbord
                cv2.imshow('Segmented Speed Sign', cropped_speed_sign)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


           # print("Geen snelheidsbord gedetecteerd in: {}".format(str(filename)))

