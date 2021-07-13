Ce projet est utilisé pour générer les fichiers `.cs` à partir des
fichiers `.tt`.

Il est nécessaire d'utiliser la version 3.1 de 'netcore'.

Pour générer un fichier `.cs` à partir du fichier `.tt` correspondant, il
faut se placer dans ce répertoire et exécuter la commande suivante:

```{.sh}
dotnet run ttfile.tt
```

Par exemple: `dotnet run ../Arcane.Axl.T4/Arcane.Axl/T4.Service/ServiceT4CaseAndStrong.tt`

