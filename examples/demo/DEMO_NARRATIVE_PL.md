# Manifest hexDAG: 15 Zasad

## Część I: Filozofia deterministyczna

1. **Najlepsze AI to if-else**: Systemy regułowe są deterministyczne, testowalne i debugowalne. Jeśli możesz rozwiązać problem regułami, nie potrzebujesz LLM-a.

2. **Programowanie z LLM-ami to programowanie stochastyczne**: Każde wywołanie modelu to eksperyment probabilistyczny.

3. **Programowanie to nie mechanika kwantowa**: Twój kod nie powinien być w superpozycji stanów. Albo działa, albo nie - nie ma "może zadziała".

4. **Walidacja to determinizm w praktyce**: Każde wejście, wyjście i przejście stanu musi mieć kontrakt. "Zwykle zwraca JSON" to nie plan na produkcję.

5. **Możesz wszystko, ale to twój problem**: Łatwo dodasz własny node, adapter, czy plugin. Jak się wywali, to też twój problem.

## Część II: Hierarchia złożoności

6. **80% AI nie potrzebuje agentów**: Większość problemów to parsowanie, klasyfikacja i routing. Deterministyczne workflow wystarczą.

7. **Z tych 20% co potrzebują agentów, 80% nie potrzebuje multi-agentów**: Jeden agent z dobrymi narzędziami rozwiąże prawie wszystko.

8. **Multi-agenty to zwykle słaba architektura**: W 80% przypadków to programista próbuje ukryć brak zrozumienia problemu za złożonością.

9. **YAML to też kod**: Jeśli nie umiesz zapisać problemu deklaratywnie, nie rozumiesz go wystarczająco.

10. **Proste się skaluje, cwane się sypie**: Najlepsza infrastruktura jest niewidoczna. Najgorsza wymaga doktoratu do debugowania.

## Część III: Prędkość i skalowalność

11. **Async-first albo przegrasz wyścig**: Blokowanie I/O w 2024 to świadome wybieranie wolności. Czekanie to strata pieniędzy.

12. **Równoległość przez analizę, nie przez modlitwę**: Deterministyczna analiza zależności automatycznie paralelizuje. Ludzie w tym są beznadziejni.

13. **Błędy szybkie i głośne**: Fail fast, fail loud. Ciche błędy zabijają produkcję i budżet.

14. **Framework robi ciężką robotę**: Retry, timeout, error handling, caching - to wszystko infrastruktura, nie twój problem.

15. **Klientów obchodzi tylko jedno**: Że działa. Szybko. Za każdym razem. Kropka.
