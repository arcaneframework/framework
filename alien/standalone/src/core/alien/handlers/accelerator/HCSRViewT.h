// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

namespace Alien
{
  template<typename MatrixT>
  class HCSRViewT
  {
  public:
    using IndexType = typename MatrixT::IndexType ;
    using ValueType = typename MatrixT::ValueType ;
    HCSRViewT(MatrixT const* parent,
            BackEnd::Memory::eType memory,
            std::size_t nrows,
            std::size_t nnz)
    : m_parent(parent)
    , m_memory(memory)
    , m_nrows(nrows)
    , m_nnz(nnz)
    {
      switch(m_memory)
      {
        case BackEnd::Memory::Device :
        {
           if constexpr (requires{m_parent->allocateDevicePointers(nrows,nnz,&m_rows,&m_ncols,&m_cols,&m_values);})
           {
             m_parent->allocateDevicePointers(nrows,nnz,&m_rows,&m_ncols,&m_cols,&m_values);
           }
           else
             throw Arccore::FatalErrorException(A_FUNCINFO, "Matrix Type doest not support allocateDevicePointers");
        }
        break ;
        case BackEnd::Memory::Host :
        default:
        {
          if constexpr (requires{m_parent->allocateHostPointers(nrows,nnz,&m_rows,&m_ncols,&m_cols,&m_values);})
          {
            m_parent->allocateHostPointers(nrows,nnz,&m_rows,&m_ncols,&m_cols,&m_values);
          }
          else
            throw Arccore::FatalErrorException(A_FUNCINFO, "Matrix Type doest not support allocateHostPointers");

        }
        break ;
      }
    }

    virtual ~HCSRViewT()
    {
      switch(m_memory)
      {
        case BackEnd::Memory::Device :
        {
           if constexpr (requires{m_parent->freeDevicePointers(m_rows,m_ncols,m_cols,m_values);})
           {
             m_parent->freeDevicePointers(m_rows,m_ncols,m_cols,m_values);
           }
        }
        break ;
        case BackEnd::Memory::Host :
        default:
        {
          if constexpr (requires{m_parent->freeHostPointers(m_rows,m_ncols,m_cols,m_values);})
          {
            m_parent->freeHostPointers(m_rows,m_ncols,m_cols,m_values);
          }
        }
        break ;
      }
    }

    MatrixT const* m_parent         = nullptr ;
    BackEnd::Memory::eType m_memory = BackEnd::Memory::Host ;
    std::size_t m_nrows             = 0 ;
    std::size_t m_nnz               = 0 ;
    IndexType* m_rows               = nullptr ;
    IndexType* m_ncols              = nullptr ;
    IndexType* m_cols               = nullptr ;
    ValueType* m_values             = nullptr ;
  };

  template<typename VectorT>
  class HVectorViewT
  {
  public:
    using IndexType = typename VectorT::IndexType ;
    using ValueType = typename VectorT::ValueType ;
    HVectorViewT(VectorT const* parent,
                 BackEnd::Memory::eType memory,
                 std::size_t nrows)
    : m_parent(parent)
    , m_memory(memory)
    , m_nrows(nrows)
    {
      switch(m_memory)
      {
        case BackEnd::Memory::Device :
        {
           if constexpr (requires{m_parent->allocateDevicePointers(nrows,&m_values);})
           {
             m_parent->allocateDevicePointers(nrows,&m_values);
           }
           else
             throw Arccore::FatalErrorException(A_FUNCINFO, "Vector Type doest not support allocateDevicePointers");
        }
        break ;
        case BackEnd::Memory::Host :
        default:
        {
          if constexpr (requires{m_parent->allocateHostPointers(nrows,&m_values);})
          {
            m_parent->allocateHostPointers(nrows,&m_values);
          }
          else
            throw Arccore::FatalErrorException(A_FUNCINFO, "Vector Type doest not support allocateHostPointers");
        }
        break ;
      }
    }

    virtual ~HVectorViewT()
    {
      switch(m_memory)
      {
        case BackEnd::Memory::Device :
        {
           if constexpr (requires{m_parent->freeDevicePointers(m_values);})
           {
             m_parent->freeDevicePointers(m_values);
           }
        }
        break ;
        case BackEnd::Memory::Host :
        default:
        {
          if constexpr (requires{m_parent->freeHostPointers(m_values);})
          {
            m_parent->freeHostPointers(m_nrows,m_values);
          }
        }
        break ;
      }
    }

    VectorT const* m_parent         = nullptr ;
    BackEnd::Memory::eType m_memory = BackEnd::Memory::Host ;
    std::size_t m_nrows             = 0 ;
    ValueType* m_values             = nullptr ;
  };
}

