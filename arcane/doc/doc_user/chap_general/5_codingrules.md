# Les règles de codage {#arcanedoc_general_codingrules}

[TOC]

Afin que les différents modules développés pour la plate-forme ARCANE 
aient une certaine homogénéité, ce document propose un ensemble de
règles de codages.


## Général {#arcanedoc_general_codingrules_general}

- le langage utilisé est le C++14
- l'encodage des fichiers est obligatoire 'UTF-8' avec le BOM au début
  du fichier. La première ligne de chaque fichier doit être comme suit:
```cpp
// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
```
- tous les commentaires utilisent la syntaxe du produit
**Doxygen** afin de pouvoir extraire directement du
code source une documentation papier ou hypertextuelle,
- tous les identifiants sont écrits en **Anglais**.
- les indentations se font avec 2 espaces. Il ne doit pas y avoir de
caractères de tabulation dans le code.

## Fichiers sources {#arcanedoc_general_codingrules_source}

- tous les fichiers sources sont formatés de la même manière et
  commencent tous par un en-tête décrivant le nom du fichier, la date
  de modification et le nom du ou des auteurs.
- vient ensuite le code proprement dit. Chaque
  identifiant doit être commenté et les fonctions doivent être séparées
  par deux lignes de commentaires. Par exemple :
```cpp
/*!
  \brief une fonction d'exemple.

  Ceci est la description longue de la fonction d'exemple.

  \param argc nombre d'arguments
  \param argv tableau des valeurs des arguments
  \return le nombre d'arguments divisés par 2
*/
int
functionExample(int argc,char** argv)
{
  return argc/2;
}
```


## Variables {#arcanedoc_general_codingrules_variable}

- les noms des variables sont toujours en **minuscules**. Si le nom 
  est composé de plusieurs mots logiques, chaque mot est séparé par le 
  caractère souligné. Par exemple :
  - \c volume
  - \c list_of_element
  - \c cells
- Pour éviter toute ambiguité, les noms pluriels sont réservés pour les
  variables de type conteneur, tous les autres noms étant au **singulier**.

## Classes {#arcanedoc_general_codingrules_classe}

- Les noms des classes (\c class) commencent par une
  majusucule et continuent par des minuscules. Si le nom est composé de
  plusieurs mots logiques, la première lettre de chaque nouveau mot est
  en majuscule. Par exemple :
  - \c Component
  - \c ComponentMng
  - \c String

- Les membres des classes, en plus de respecter les mêmes conventions
  que n'importe quelle variable, seront toujours préfixés par les deux
  caractères <tt>m_</tt>. Par exemple :
  - \c m_volume
  - \c m_list_of_element

## Méthodes et fonctions {#arcanedoc_general_codingrules_method}

Dans ce qui suit, on utilisera le mot fonction pour
désigner à la fois les fonctions et les méthodes de classe.

- les noms des fonctions sont toujours en <b>minuscules</b>. Si le
  nom est composé de plusieurs mots logiques, la première lettre de
  chaque nouveau mot est en majuscule. Par exemple :
   - \c numberOfElement()
   - \c assign()
- si le nom de la méthode correspond à la notion de propriété (c'est à dire
  équivalent sémantiquement à un champ de la classe) de nom \a value, l'accesseur doit
  être le nom de la propriété (\a value()) et la méthode pour
  changer la valeur doit être \a setValue(). <strong>L'accesseur ne doit
  pas commencer par `get`</strong>. Si la propriété est booléenne, il est possible de
  préfixer l'accesseur par \a is. Par exemple \a isEmpty().

- Pour éviter toute ambiguité, tous les noms sont au <b>singulier</b>
- La définition des fonctions se fait sur au moins deux lignes:
  - la première comprend le type de retour et éventuellement
    le nom de la classe s'il s'agit d'une méthode.
  - la deuxième comprend obligatoirement le nom de la fonction.
  - viennent ensuite la liste des arguments sur la deuxième ligne et
    les suivantes.
- L'accolade ouvrant le corps de la fonction et celle le fermant
doivent être sur une ligne séparée :
```cpp
int
function1(int argc,char** argv)
{
  return argc/2;
}
```


## Exemple {#arcanedoc_general_codingrules_example}

```cpp
/*!
 * \brief Tableau constant d'un type \a T.

 Cette classe encapsule un tableau C constant standard (pointeur) et son nombre
 d'éléments. L'accès à ses éléments se fait par l'opérateur operator[]().
 La méthode base() permet d'obtenir le pointeur du tableau pour le passer
 aux fonctions C standard.

 L'instance conserve juste un pointeur sur le début du tableau C et ne fait
 aucune gestion mémoire. Le développeur doit s'assurer que le pointeur
 reste valide tant que l'instance existe.

 Les éléments du tableau ne peuvent pas être modifiés.

 En mode débug, une vérification de débordement est effectuée lors de l'accès
 à l'opérateur operator[]().
 */
template<typename T>
class ConstCArrayT
{
 private:

 protected:

 public:
	
  //! Type des éléments du tableau
  typedef T value_type;
  //! Type de l'itérateur constant sur un élément du tableau
  typedef const value_type * key_restrict const_iterator;
  //! Type pointeur constant d'un élément du tableau
  typedef const value_type * key_restrict const_pointer;
  //! Type référence constante d'un élément du tableau
  typedef const value_type& const_reference;
  //! Type indexant le tableau
  typedef Integer size_type;
  //! Type d'une distance entre itérateur éléments du tableau
  typedef ptrdiff_t difference_type;

  //! Type d'un itérateur constant sur tout le tableau
  typedef ConstIterT< ConstCArrayT<T> > const_iter;

 public:

  //! Construit un tableau vide.
  ConstCArrayT() : m_size(0), m_ptr(0) {}
  //! Construit un tableau avec \a s élément
  explicit ConstCArrayT(Integer s,const T* ptr)
  : m_size(s), m_ptr(ptr) {}
  /*! \brief Constructeur par copie.
   * \warning Seul le pointeur est copié. Aucune copie mémoire n'est effectuée.
   */
  ConstCArrayT(const ConstCArrayT<T>& from)
  : m_size(from.m_size), m_ptr(from.m_ptr) {}
  /*! \brief Constructeur par copie.
   * \warning Seul le pointeur est copié. Aucune copie mémoire n'est effectuée.
   */
  ConstCArrayT(const CArrayBaseT<T>& from)
  : m_size(from.size()), m_ptr(from.begin())
    {
    }

  /*! \brief Opérateur de recopie.
   * \warning Seul le pointeur est copié. Aucune copie mémoire n'est effectuée.
   */
  const ConstCArrayT<T>& operator=(const ConstCArrayT<T>& from)
    { m_size=from.m_size; m_ptr=from.m_ptr; return *this; }
	
  /*! \brief Opérateur de recopie.
   * \warning Seul le pointeur est copié. Aucune copie mémoire n'est effectuée.
   */
  const ConstCArrayT<T>& operator=(const CArrayBaseT<T>& from)
    {
      m_size = from.size();
      m_ptr  = from.begin();
      return (*this);
    }

 public:

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  const T& operator[](Integer i) const
    {
      return m_ptr[i];
    }

  //! Nombre d'éléments du tableau
  inline Integer size() const { return m_size; }
  //! Iterateur sur le premier élément du tableau
  inline const_iterator begin() const { return m_ptr; }
  //! Iterateur sur le premier élément après la fin du tableau
  inline const_iterator end() const { return m_ptr+m_size; }
  //! \a true si le tableau est vide (size()==0)
  inline bool empty() const { return m_size==0; }

  //! Pointeur sur le début du tableau.
  inline const T* base() const { return m_ptr; }

 protected:

 private:

  Integer m_size; //!< Nombre d'éléments 
  const T* m_ptr; //!< Pointeur sur le début du tableau
};
```



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_general_traces
</span>
<!-- <span class="next_section_button">
\ref arcanedoc_getting_started_basicstruct
</span> -->
</div>
