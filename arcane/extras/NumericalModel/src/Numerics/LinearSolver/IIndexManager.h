// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef IINDEX_MANAGER_H
#define IINDEX_MANAGER_H

#include <arcane/IVariable.h>
#include <arcane/Item.h>
#include <arcane/utils/ObjectImpl.h>
#include <arcane/utils/Array.h>
#include <arcane/utils/AutoRef.h>
#include <arcane/IItemFamily.h>
#include <arcane/utils/FatalErrorException.h>
#include <arcane/utils/ITraceMng.h>
#include <arcane/ArcaneVersion.h>

using namespace Arcane;

/** 
 * Voir \ref secLinearSolver "Description des services de solveurs
 * linéaires" pour plus de détails
 *
 * \todo Pour gérer les inconnues vectoriel, mettre un argument additionnel qui
 * est le sous-indice (vectoriel) avec une valeur par défaut de 0 (ou
 * bien un proto différent)
 */
class IIndexManager {
public:
  typedef Integer EntryIndex;
  typedef Integer EquationIndex;


  //! Méthode standard d'accès aux variables
  enum eVariableAccess 
    { 
      Undefined,   //! Non défini
      Direct,      //! Copie à l'initialisation et à la mise à jour
      Incremental  //! Initialisation à zéro et mise à jour par incrémentation
    };

  //! Interface d'initialisation d'une \a Entry
  /*! De l'espace des variables vers sa représentation dans le système */
  class Initializer :
    public ObjectImpl
  {
  public:
    virtual void init(const ConstArrayView<Item> & items, Array<Real> & values) = 0;
  };

  //! Définit le gestionnaire de trace
  virtual void setTraceMng(ITraceMng * traceMng) = 0;

  //! Interface de mise à jour d'une \a Entry
  /*! De sa représentation dans le système vers l'espace des variables */
  class Updater :
    public ObjectImpl
  {
  public:
    virtual void update(const ConstArrayView<Item> & items, const Array<Real> & values) = 0;
  };

  
  //! Interface d'implémentation de \a Entry
  class EntryImpl 
  {
  public:
    //! Destructeur
    virtual ~EntryImpl() { }
    //! Retourne la liste des Index de l'Entry
    virtual ConstArrayView<Integer> getIndex() const = 0;
    //! Positionne l'initialiseur
    virtual void setInitializer(Initializer * initializer) = 0;
    //! Teste le besoin d'une initialisation 
    /*! ie initialiseur existe */
    virtual bool needInit() const = 0;
    //! Remplit les valeurs à d'initialiser l'Entry 
    /*! Usuellement pour la valeur initiale d'un système linéaire */
    virtual void initValues(Array<Real> & values) = 0;
    //! Positionne l'updater
    virtual void setUpdater(Updater * updater) = 0;
    //! Teste si une mise à jour est nécessaire
    /*! ie updater existe */
    virtual bool needUpdate() const = 0;
    //! Retourne les valeurs de mise à jour de l'Entry
    virtual void updateValues(const Array<Real> & values) = 0;
    //! Retourne la variable associée à l'Entry
    /*! NULL si n'est pas associé à une variable */
    virtual IVariable * getVariable() const = 0;
    //! Retourne le nom associé à l'Entry
    virtual String getName() const = 0;
    //! Retourne le type de support de l'Entry
    virtual eItemKind getKind() const = 0;
    //! Retourne la famille d'item indexant cette entrée
    virtual const IItemFamily * getItemFamily() const = 0;
    //! Ajout d'un tag
    virtual void addTag(const String &tagname, const String &tagvalue) = 0;
    //! Suppression d'un tag
    virtual void removeTag(const String &tagname) = 0;
    //! Test d'existance d'un tag
    virtual bool hasTag(const String &tagname) = 0;
    //! Lecture d'un tag
    virtual String tagValue(const String & tagname) = 0;
  };


  //! Classe de représentation des Entry
  /*! Cette classe est un proxy; sa copie est donc peu couteuse et son
   *  implémentation variable suivant le contexte
   */
  class Entry
  {
  protected:
    //! Implémentation de ce type d'entrée
    EntryImpl * m_impl;
  public:
    //! Constructeur par défaut
    Entry() : m_impl(NULL) { }

    //! Constructeur par copie
    Entry(const Entry & en) : m_impl(en.m_impl) { }

    //! Constructeur
    Entry(EntryImpl * impl) : m_impl(impl) { }

    //! Opérateur de copie
    Entry & operator=(const Entry & en) { 
      if (this != &en)
        m_impl = en.m_impl;
      return *this;      
    }
    
    //! Accès interne à l'implementation
    EntryImpl * internal() const { return m_impl; }    

    //! Indique si l'entrée est définie
    bool null() const { return m_impl == NULL; }

    //! Ensemble des indices gérés par cette entrée
    ConstArrayView<Integer> getIndex() const { return m_impl->getIndex(); }

    //@{ @name Gestion des mises à jour
    //! Définition de l'"Initializer" associé
    void setInitializer(Initializer * init) { m_impl->setInitializer(init); }
    
    //! Vérifie le besoin d'initialisation
    bool needInit() const { return m_impl->needInit(); }

    //! Initialisation à partir d'un ensemble de valeurs
    void initValues(Array<Real> & values) { m_impl->initValues(values); }

    //! Définition de l'"Updater" associé
    void setUpdater(Updater * updater) { m_impl->setUpdater(updater); }

    //! Vérifie le besoin de mise à jour
    bool needUpdate() const { return m_impl->needUpdate(); }

    //! Mise à jour d'un groupe de valeurs
    void updateValues(const Array<Real> & values) { m_impl->updateValues(values); }
    // @}

    //! Référence à la variable gérée
    /*! Retourne NULL si ce n'est une variable qui est gérée */
    IVariable * getVariable() const { return m_impl->getVariable(); }

    //! Nom de l'entrée
    /*! Si l'entrée est associée à une variable retourne le nom de la variable */
    String getName() const { return m_impl->getName(); }

    //! Support de l'entrée (en terme d'item)
    eItemKind getKind() const { return m_impl->getKind(); }

    //! Retourne la famille d'item indexant cette entrée
    const IItemFamily * getItemFamily() const { return m_impl->getItemFamily(); }

    //@{ @name Gestion des tags
    //! Ajout d'un tag
    void addTag(const String &tagname, const String &tagvalue) { return m_impl->addTag(tagname,tagvalue); }

    //! Suppression d'un tag
    void removeTag(const String &tagname) { return m_impl->removeTag(tagname); }

    //! Test d'existance d'un tag
    bool hasTag(const String &tagname) { return m_impl->hasTag(tagname); }

    //! Acces en lecture à un tag
    String tagValue(const String & tagname) { return m_impl->tagValue(tagname); }
    //@}
  };

  
  //! Interface d'implémentation de \a Equation
  class EquationImpl {
  public:
    //! Destructeur
    virtual ~EquationImpl() { }
    //! Retourne l'entrée associé
    /*! Si elle existe */
    virtual Entry getEntry() const = 0;
    //! Retourne le nom associé à l'Equation
    virtual String getName() const = 0;
    //! Retourne le type de support de l'Equation
    virtual eItemKind getKind() const = 0;
    //! Retourne la famille d'item indexant cette équation
    virtual const IItemFamily * getItemFamily() const = 0;
  };


  //! Classe de représentation des Entry
  /*! Cette classe est un proxy; sa copie est donc peu couteuse et son
   *  implémentation variable suivant le contexte
   */
  class Equation {
  protected:
    //! Implémentation de ce type d'entrée
    EquationImpl * m_impl;
  public:
    //! Constructeur par défaut
    Equation() : m_impl(NULL) { }

    //! Constructeur par copie
    Equation(const Equation & eq) : m_impl(eq.m_impl) { }

    //! Constructeur
    Equation(EquationImpl * impl) : m_impl(impl) { }

    //! Opérateur de copie
    Equation & operator=(const Equation & eq) { 
      if (this != &eq)
        m_impl = eq.m_impl;
      return *this;      
    }

    //! Accès interne à l'implementation
    EquationImpl * internal() const { return m_impl; }

    //! Référence à l'entrée associée
    /*! Si elle existe */
    Entry getEntry() const { return m_impl->getEntry(); }

    //! Nom de l'équation
    String getName() const { return m_impl->getName(); }

    //! Support de l'entrée (en terme d'item)
    eItemKind getKind() const { return m_impl->getKind(); }

    //! Retourne la famille d'item indexant cette équation
    const IItemFamily * getItemFamily() const { return m_impl->getItemFamily(); }
  };

  //! Interface d'implementation de \a EntryEnumerator
  class EntryEnumeratorImpl : 
    public ObjectImpl {
  public:
    virtual void moveNext() = 0;
    virtual bool hasNext() const = 0;
    virtual EntryImpl * get() const = 0;
  };


  //! Classe d'énumération des \a Entry connues
  /*! Classe de type proxy; la copie est peu couteuse et l'implémentation variable */
  class EntryEnumerator {
  protected:
      AutoRefT<EntryEnumeratorImpl> m_impl;
  public:
    //! Constructeur par copie
    EntryEnumerator(const EntryEnumerator & e) : m_impl(e.m_impl) { }
    //! Constructeur par consultation de l'\a IndexManager
    EntryEnumerator(IIndexManager * manager) : m_impl(manager->enumerateEntry().m_impl) { }
    //! Constructeur par implémentation
    EntryEnumerator(EntryEnumeratorImpl * impl) : m_impl(impl) { }
    //! Avance l'énumérateur
    void operator++() { m_impl->moveNext(); }
    //! Teste l'existence d'un élément suivant
    bool hasNext() const { return m_impl->hasNext(); }
    //! Déréférencement
    Entry operator*() const { return m_impl->get(); }
    //! Déréférencement indirect
    EntryImpl * operator->() const { return m_impl->get(); }
  };


 public:
  //! Constructeur par défaut
  IIndexManager() 
  { 
    ;
  }

  //! Destructeur
  virtual ~IIndexManager()
  {
    ;
  }
  
  virtual bool isReady() = 0 ;
  
  //! Initialisation
  virtual void init() = 0;

  //! Construit une nouvelle entrée abstraite
  virtual Entry buildAbstractEntry(const String name, const IItemFamily * itemFamily) = 0;

  //! Construit une entrée associée à une variable
  virtual Entry buildVariableEntry(IVariable * ivar, const eVariableAccess mode) = 0;

  //! Retourne l'entrée associée à un nom
  /*! Valable aussi pour un accès via un nom de variable */
  virtual Entry getEntry(const String name) const = 0;

  //! Retourne l'entrée associée à une variable
  virtual Entry getVariableEntry(IVariable * ivar) const = 0;

  //! Construit une équation indépendante
  /*! Ceci implique une politique de placement de l'équation dans le
   *  système mal définie
   */
  virtual Equation buildEquation(const String name, const IItemFamily * itemFamily) = 0;

  //! Construit une équation dominée par une entrée
  /*! L'indexation de cette équation sera associée à celle de l'entrée
   *  fournit
   */
  virtual Equation buildEquation(const String name, const Entry & entry) = 0;

  //! Retourne l'équation via son nom
  virtual Equation getEquation(const String name) = 0;

  //! Demande d'indexation d'une entrée (colonne)
  /*! Utilisable uniquement avant prepare */
  virtual EntryIndex defineEntryIndex(const Entry & entry, const Item & item) = 0;

  //! Demande d'indexation d'une entrée (colonnes)
  /*! Utilisable uniquement avant prepare */
  virtual void defineEntryIndex(const Entry & entry, const ItemGroup & itemGroup) = 0;

  //! Demande d'indexation d'une équation (ligne)
  /*! Utilisable uniquement avant prepare */
  virtual EquationIndex defineEquationIndex(const Equation & entry, const Item & item) = 0;

  //! Demande d'indexation d'une équation (lignes)
  /*! Utilisable uniquement avant prepare */
  virtual void defineEquationIndex(const Equation & entry, const ItemGroup & itemGroup) = 0;

  //! Préparation : fixe l'indexation (fin des définitions) 
  virtual void prepare() = 0;

  //! Consultation d'indexation d'une entrée (après prepare) 
  virtual Integer getEntryIndex(const Entry & entry, const Item & item) const = 0;

  //! Consultation de réindexation (après prepare) 
  virtual Integer getEquationIndex(const Equation & equation, const Item & item) const = 0;

  //! Consultation vectorielle d'indexation d'une entrée (après prepare) 
  virtual void getEntryIndex(const Entry & entry, const ItemVectorView & items, ArrayView<Integer> indexes) const = 0;

  //! Consultation vectorielle de réindexation (après prepare) 
  virtual void getEquationIndex(const Equation & equation, const ItemVectorView & items, ArrayView<Integer> indexes) const = 0;

  //! Translation d'un EntryIndex en indice de colonne
  virtual Integer getEntryIndex(const EntryIndex & index) const = 0;

  //! Translation d'un EquationIndex en indice de ligne
  virtual Integer getEquationIndex(const EquationIndex & index) const = 0;

  //! Fournit une table de translation indexé par les items
  virtual IntegerArray getEntryIndexes(const Entry & entry) const = 0;

  //! Fournit une table de translation indexé par les items
  virtual IntegerArray getEquationIndexes(const Equation & equation) const = 0;

  //! Décrit si l'ordre des getEntryIndexes et getEquationIndexes associés sont compatibles
  /*! Dans ce cas, juste getEntryIndexes est suffisant pour en faire un cache local */
  virtual bool hasCompatibleIndexOrder(const Entry & entry, const Equation & equation) const = 0;

  //! Statistiques d'indexation
  /*! Uniquement valide après \a prepare */
  virtual void stats(Integer & totalSize,
		     Integer & minLocalIndex,
		     Integer & localSize) const = 0;

  //! Construction d'un enumerateur sur les \a Entry
  virtual EntryEnumerator enumerateEntry() = 0;
};

#endif /* IINDEX_MANAGER_H */
