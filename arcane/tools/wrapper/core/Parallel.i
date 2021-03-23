// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
namespace Arcane
{
  namespace Parallel
  {
    // Reporte les définitions de Arccore::MessagePassing
    enum eReduceType
    {
      ReduceMin,
      ReduceMax,
      ReduceSum
    };
    enum eWaitType
    {
      WaitAll,
      WaitSome,
      WaitSomeNonBlocking
    };
  }
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%define SWIG_ARCANE_PARALLEL_DISPATCH(Type)
  virtual void broadcast(ArrayView<Type> send_buf,Integer sub_domain) =0;
  virtual void allGather(ConstArrayView<Type> send_buf,ArrayView<Type> recv_buf) =0;
  virtual void allGatherVariable(ConstArrayView<Type> send_buf,Array<Type>& recv_buf) =0;
  virtual void scatterVariable(ConstArrayView<Type> send_buf,ArrayView<Type> recv_buf,Integer root) =0;
  virtual void allToAll(ConstArrayView<Type> send_buf,ArrayView<Type> recv_buf,Integer count) =0;
  virtual void allToAllVariable(ConstArrayView<Type> send_buf,Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index,ArrayView<Type> recv_buf,
                                Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) =0;
  virtual Parallel::Request send(ConstArrayView<Type> send_buffer,Integer proc,bool is_blocked) =0;
  virtual Parallel::Request recv(ArrayView<Type> recv_buffer,Integer proc,bool is_blocked) =0;
  virtual void send(ConstArrayView<Type> send_buffer,Integer proc) =0;
  virtual void recv(ArrayView<Type> recv_buffer,Integer proc) =0;
  virtual void sendRecv(ConstArrayView<Type> send_buffer,ArrayView<Type> recv_buffer,Integer proc) =0;
  virtual Type reduce(Parallel::eReduceType op,Type send_buf) =0;
  virtual void reduce(Parallel::eReduceType op,ArrayView<Type> send_buf) =0;
  virtual void scan(Parallel::eReduceType op,ArrayView<Type> send_buf) =0;
  virtual void computeMinMaxSum(Type val,Type& min_val,Type& max_val,Type& sum_val,
                                Int32& min_rank,
                                Int32& max_rank) =0;
%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
  class IParallelMng
  {
   protected:
    virtual ~IParallelMng(){}
   public:
    SWIG_ARCANE_PARALLEL_DISPATCH(Byte);
    SWIG_ARCANE_PARALLEL_DISPATCH(SByte);
    //SWIG_ARCANE_PARALLEL_DISPATCH(Int16);
    SWIG_ARCANE_PARALLEL_DISPATCH(Int32);
    SWIG_ARCANE_PARALLEL_DISPATCH(Int64);
    //SWIG_ARCANE_PARALLEL_DISPATCH(UInt16);
    SWIG_ARCANE_PARALLEL_DISPATCH(UInt32);
    SWIG_ARCANE_PARALLEL_DISPATCH(UInt64);
    SWIG_ARCANE_PARALLEL_DISPATCH(Real);
    //SWIG_ARCANE_PARALLEL_DISPATCH(Real2);
    SWIG_ARCANE_PARALLEL_DISPATCH(Real3);
    //SWIG_ARCANE_PARALLEL_DISPATCH(Real2x2);
    //SWIG_ARCANE_PARALLEL_DISPATCH(Real3x3);
  };        
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
