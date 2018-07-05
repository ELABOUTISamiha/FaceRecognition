
from ImageDbManager import ImageDbManager
from ProcessingModule import *


if __name__ == "__main__":
    print("-------   PROGRAMME DE RECONNAISSANCE FACIALE DEVELOPPE PAR ELABOUTI SAMIHA   -------")
    print("Entrez :")
    print("\t\t1 )- Pour lancer la partie acquisition.")
    print("\t\t2 )- Pour lancer la partie reconnaissance.")
    print("\t\tautre )- Pour quitter l'application.")

    choix = raw_input()

    if choix == '1' :
        print("-------   MODULE D'AQUISITION D'IMAGES BY  SAMIHA  -------")
        print("Entrez l'ID de la nouvelle personne:   ")
        id_r = int(raw_input())

        manager = ImageDbManager(id=id_r)
        manager.run()
        # IMAGES SONT MAINTENANT AQUISES

    if choix == '2' :
        # Lancer la reconnaissance
        train_recognizer()
        run_recognizer()

    else :
        pass



