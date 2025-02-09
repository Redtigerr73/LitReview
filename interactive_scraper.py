#!/usr/bin/env python3
import sys
import datetime
from scraper import ScholarScraper

def print_welcome():
    """Affiche le message de bienvenue"""
    print("\n" + "="*50)
    print("  Bienvenue dans le Google Scholar Scraper")
    print("  Version 2.0 - Février 2025")
    print("="*50 + "\n")
    print("Ce programme vous permet de rechercher des articles scientifiques")
    print("sur Google Scholar et de sauvegarder les résultats au format CSV.\n")

def get_search_queries():
    """Obtient les requêtes de recherche de l'utilisateur"""
    print("\n" + "-"*50)
    print("ÉTAPE 1: Définition des mots-clés de recherche")
    print("-"*50)
    print("Entrez vos mots-clés de recherche, séparés par des virgules.")
    print("Exemple: deep learning, neural networks, machine learning\n")
    
    while True:
        input_text = input("Mots-clés > ").strip()
        if not input_text:
            print("⚠️  Veuillez entrer au moins un mot-clé.")
            continue
        
        queries = [q.strip() for q in input_text.split(',') if q.strip()]
        if not queries:
            print("⚠️  Format invalide. Utilisez des virgules pour séparer les mots-clés.")
            continue
            
        print(f"\n✅ {len(queries)} requête(s) enregistrée(s):")
        for i, q in enumerate(queries, 1):
            print(f"   {i}. {q}")
        return queries

def get_article_count():
    """Obtient le nombre d'articles souhaités"""
    print("\n" + "-"*50)
    print("ÉTAPE 2: Nombre d'articles à rechercher")
    print("-"*50)
    print("Combien d'articles souhaitez-vous rechercher au total ?")
    print("(Min: 10, Max: 10000, par tranches de 10)\n")
    
    while True:
        try:
            count = int(input("Nombre d'articles > "))
            if 10 <= count <= 10000:
                pages = (count + 9) // 10  # Arrondi supérieur
                print(f"\n✅ Recherche de {count} articles ({pages} pages)")
                return pages
            print("⚠️  Le nombre doit être entre 10 et 10000.")
        except ValueError:
            print("⚠️  Veuillez entrer un nombre valide.")

def generate_output_filename(queries):
    """Génère le nom du fichier de sortie"""
    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")
    query_str = queries[0].lower().replace(" ", "_")[:30]
    if len(queries) > 1:
        query_str += f"_plus_{len(queries)-1}"
    return f"data/csv/scholar_{date_str}_{query_str}.csv"

def main():
    try:
        # Affichage du message de bienvenue
        print_welcome()
        
        # Récupération des requêtes
        queries = get_search_queries()
        
        # Récupération du nombre d'articles
        max_pages = get_article_count()
        
        # Génération du nom de fichier
        outfile = generate_output_filename(queries)
        
        # Résumé et confirmation
        print("\n" + "-"*50)
        print("RÉSUMÉ DE LA RECHERCHE")
        print("-"*50)
        print(f"• Nombre de requêtes: {len(queries)}")
        print(f"• Pages à scraper: {max_pages}")
        print(f"• Articles attendus: ~{max_pages * 10}")
        print(f"• Fichier de sortie: {outfile}")
        print("\nLe scraping peut prendre plusieurs minutes.")
        
        confirm = input("\nDémarrer la recherche ? (o/n) > ").lower()
        if confirm != 'o':
            print("\n❌ Opération annulée")
            sys.exit(0)
        
        # Initialisation et exécution du scraper
        print("\n" + "-"*50)
        print("PROGRESSION DU SCRAPING")
        print("-"*50)
        
        scraper = ScholarScraper(outfile)
        total_operations = len(queries) * max_pages
        current_operation = 0
        
        def progress_callback():
            nonlocal current_operation
            current_operation += 1
            progress = (current_operation / total_operations) * 100
            print(f"\rProgression: {progress:.1f}% ", end="", flush=True)
        
        scraper.run(queries, max_pages, start_num=0, progress_callback=progress_callback)
        
        print("\n\n✅ Scraping terminé avec succès!")
        print(f"📊 Résultats sauvegardés dans: {outfile}")
        
    except KeyboardInterrupt:
        print("\n\n❌ Opération interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Erreur: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
