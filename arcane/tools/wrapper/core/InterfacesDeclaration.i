// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
// Déclaration des interfaces.
// Les interfaces déclarées ici doivent ensuite avoir le fichier d'en-tête associé
// ajouté dans le fichier 'InterfacesInclude.h'.

// Déclaration des interfaces.
// Pour l'instant cela ne fonctionne que en C#.
#if defined(SWIGCSHARP)
ARCANE_DECLARE_INTERFACE(Arcane,IDataReader)
ARCANE_DECLARE_INTERFACE(Arcane,IDataWriter)
ARCANE_DECLARE_INTERFACE(Arcane,IUnitTest)
ARCANE_DECLARE_INTERFACE(Arcane,IMeshReader)
ARCANE_DECLARE_INTERFACE(Arcane,ICaseFunctionProvider)
ARCANE_DECLARE_INTERFACE(Arcane,ICaseFunctionDotNetProvider)
ARCANE_DECLARE_INTERFACE(Arcane,ICheckpointWriter)
ARCANE_DECLARE_INTERFACE(Arcane,ICheckpointReader)

ARCANE_DECLARE_INTERFACE(Arcane,IVariableReader)
ARCANE_DECLARE_INTERFACE(Arcane,IDirectExecution)
ARCANE_DECLARE_INTERFACE(Arcane,ITimeHistoryCurveWriter2)
ARCANE_DECLARE_INTERFACE(Arcane,IPostProcessorWriter)
ARCANE_DECLARE_INTERFACE(Arcane,IExternalPlugin)
#endif
