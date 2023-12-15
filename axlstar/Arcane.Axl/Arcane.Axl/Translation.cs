//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;

namespace Arcane.Axl
{
  public class Translation
  {
    //! Titre pour le module pour la doc. 0 = nom du module, 1 = nom de l'élément xml des options du module
    public string ModuleTitle = "'{0}' Module (<{1}>)";
    //! Titre pour le service pour la doc. 0 = nom du service.
    public string ServiceTitle = "'{0}' Service";

    public string ListOfModules = "List of modules";
    public string ListOfServices = "List of services";
    public string ListOfInterfaces = "List of interfaces";
    public string DetailedListOfOptions = "Detailed list of options";
    public string ModuleAndServices = "Modules and services";
    public string KeywordsIndex = "Keywords index";
    public string ServiceInterfaces = "Service interfaces";
    public string Implementations = "Implementation(s)";
    public string ServiceUsed = "Service used in";

    public string Lang { get; private set; }

    public Translation(string language)
    {
      Lang = language;
      if (language == "fr")
      {
        ModuleTitle = "Module '{0}' <{1}>";
        ServiceTitle = "Service '{0}'";
        ListOfModules = "Liste des modules";
        ListOfServices = "Liste des services";
        ListOfInterfaces = "Liste des interfaces";
        DetailedListOfOptions = "Liste détaillée des options";
        ModuleAndServices = "Listes des modules et des services";
        KeywordsIndex = "Index des mots clés";
        ServiceInterfaces = "Interface des services";
        Implementations = "Implémentation(s)";
        ServiceUsed = "Service utilisé dans";
      }
    }
  }
}

