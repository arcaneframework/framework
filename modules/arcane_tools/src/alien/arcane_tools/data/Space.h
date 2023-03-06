#pragma once



#include <alien/data/Space.h>
#include <alien/arcane_tools/IIndexManager.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace ArcaneTools {


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Space final
  : public Alien::ISpace
{
 public:
  
  Space() {}
  
  Space(IIndexManager* index_mng, Arccore::String name = "Undefined");

  Space(IIndexManager* index_mng, Integer block_size, Arccore::String name = "Undefined");
  
  Space(const Space& s) 
    : m_internal(s.m_internal) 
    , m_index_mng(s.m_index_mng) {}
    
  Space(Space&& s)
    : m_internal(s.m_internal)
    , m_index_mng(s.m_index_mng) {}
    
  ~Space() {}
    
  Space& operator=(const Space& s)
    {
      m_internal = s.m_internal;
      m_index_mng = s.m_index_mng;
      return *this;
    }
  Space& operator=(Space&& s)
    {
      m_internal = s.m_internal;
      m_index_mng = s.m_index_mng;
      return *this;
    }
    
  //! Comparaison entre deux espaces
  /*! Deux espaces sont ��gaux si c'est le m��me */
  bool operator==(const Alien::ISpace& space) const
  {
    return *m_internal == space;
  }
  bool operator!=(const Alien::ISpace& space) const
  {
    return *m_internal != space;
  }
    
  //! Taille de l'espace (SD: i.e. taille de la base, dimension ou cardinalit��)
  Integer size() const { return m_internal->size(); }
    
  //! Nom de l'espace
  const String& name() const { return m_internal->name(); }
    
  //! Ajout de champs, ie indices par label
  void setField(String label, const UniqueArray<Integer>& indices)
  {
    m_internal->setField(label, indices);
  }
    
  //! Nombre de champs
  Integer nbField() const { return m_internal->nbField(); }
    
  //! Label du ieme champ
  String fieldLabel(Integer i) const { return m_internal->fieldLabel(i); }
    
  //! Retrouve les indices du ieme champ
  const UniqueArray<Integer>& field(Integer i) const { return m_internal->field(i); }
    
  //! Retrouve les indices du champ �� partir du label
  const UniqueArray<Integer>& field(String label) const { return m_internal->field(label); }
    
  //! Cheap clone
  std::shared_ptr<Alien::ISpace> clone() const 
  {
    return std::make_shared<Space>(*this); 
  }
    
  //! Index manager
  IIndexManager* indexManager() const { return m_index_mng; }

 private:
  void _init();
  
 private:
    
  std::shared_ptr<Alien::Space> m_internal;
    
  IIndexManager* m_index_mng = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

