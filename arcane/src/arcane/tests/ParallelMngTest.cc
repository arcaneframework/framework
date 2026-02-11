// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelMngTest.cc                                          (C) 2000-2025 */
/*                                                                           */
/* Test des opérations de base du parallèlisme.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/ValueChecker.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Array3View.h"
#include "arcane/utils/Array2.h"
#include "arcane/utils/NumericTypes.h"
#include "arcane/utils/HPReal.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/core/FactoryService.h"
#include "arcane/core/AbstractService.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IDirectExecution.h"
#include "arcane/core/SerializeBuffer.h"
#include "arcane/core/IParallelExchanger.h"
#include "arcane/core/ISerializeMessage.h"
#include "arcane/core/ISerializeMessageList.h"
#include "arcane/core/IParallelTopology.h"
#include "arcane/core/IParallelNonBlockingCollective.h"
#include "arcane/core/MachineMemoryWindow.h"
#include "arcane/core/DynamicMachineMemoryWindow.h"
#include "arcane/core/DynamicMachineMemoryWindowMemoryAllocator.h"
#include "arcane/core/ParallelMngUtils.h"
#include "arcane/core/VariableBuildInfo.h"
#include "arcane/core/internal/SerializeMessage.h"

#include "arcane/tests/ArcaneTestGlobal.h"

#include "arcane/parallel/IRequestList.h"

#include "arccore/message_passing/Messages.h"

#include <cstdint>
#include <thread>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Arcane::Parallel;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Test des opérations de base du IParallelMng
 */
class ParallelMngTest
: public TraceAccessor
{
  class SerializerTestValues;
 public:

  ParallelMngTest(IParallelMng* pm,const String& test_name);

 public:

  //! Exécute l'opération du service
  void execute();

 private:

  IParallelMng* m_parallel_mng = nullptr;
  bool m_verbose = false;
  bool m_test_broadcast_serializer;
  String m_test_name;
  Int32 m_nb_done_test = 0;

 private:

  void _testSendRecvNonBlocking3();
  void _testSendRecvNonBlockingSome(Integer nb_message,Integer message_size,bool is_non_blocking);
  void _testSerialize1();
  void _doTestSerializerWithMessageInfo(bool use_wait_all);
  void _testSerializerWithMessageInfo();
  void _testSerializeSize(Integer nb_value);
  void _doTestSerializeSize(bool is_non_blocking);
  void _testNonBlockingSerializeSize(Integer nb_value);
  void _doTestSerializeMessageList();
  void _doTestSerializeMessageList2(eWaitType wait_mode);
  void _testSerializeMessageList(Integer nb_value,eWaitType wait_mode);
  void _testSerializerWithMessageInfo(Integer nb_value,bool use_wait);
  template<typename DataType> void _testParallelBasic(DataType data);
  void _testReduce2();
  void _launchTest(const String& test_name,void (ParallelMngTest::*func)());
  void _testBarrier();
  void _testProcessMessages();
  void _testBroadcastSerializer();
  void _testBroadcastSerializer2(Integer n);
  void _testTopology();
  void _testStandardCalls();
  void _testNamedBarrier();
  void _testBroadcastStringAndMemoryBuffer();
  void _testBroadcastStringAndMemoryBuffer2(const String& wanted_str);
  void _testProbeSerialize(Integer nb_value,bool use_one_message);
  void _testProcessMessages(const ParallelExchangerOptions* exchange_options);
  void _testMachineMemoryWindow();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelMngTest::
ParallelMngTest(IParallelMng* pm,const String& test_name)
: TraceAccessor(pm->traceMng())
, m_parallel_mng(pm)
, m_verbose(false)
, m_test_broadcast_serializer(true)
, m_test_name(test_name)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
execute()
{
#if 0
  {
    CriticalSection cs(psm->threadMng());
    IParallelMng* pm = psm->createParallelMng();
    m_parallel_mng = pm;
    pm->setBaseObject(m_application);
  }
#endif

  info() << "EXEC DIRECT my_rank=" << m_parallel_mng->commRank()
         << " comm_size=" << m_parallel_mng->commSize()
         << " test_name=" << m_test_name;

  _launchTest("serialize",&ParallelMngTest::_testSerialize1);
  _launchTest("serializer_with_message_info",
              &ParallelMngTest::_testSerializerWithMessageInfo);

  _launchTest("serialize_message_list",&ParallelMngTest::_doTestSerializeMessageList);
  _launchTest("datatype",&ParallelMngTest::_testStandardCalls);

  _launchTest("named_barrier",&ParallelMngTest::_testNamedBarrier);
  _launchTest("broadcast_string",&ParallelMngTest::_testBroadcastStringAndMemoryBuffer);

  //TODO ajouter tests broadcast avec n'importe quel proc comme destinataire
  //TODO ajouter tests send/recv avec n'importe quel proc comme destinataire

  _launchTest("barrier",&ParallelMngTest::_testBarrier);
  _launchTest("process_messages",&ParallelMngTest::_testProcessMessages);

  _launchTest("send_receive_nb3",&ParallelMngTest::_testSendRecvNonBlocking3);

  _launchTest("reduce2",&ParallelMngTest::_testReduce2);

  if (m_test_broadcast_serializer){
    _launchTest("broadcast_serializer",&ParallelMngTest::_testBroadcastSerializer);
  }
  _launchTest("topology",&ParallelMngTest::_testTopology);

  _launchTest("machine_window", &ParallelMngTest::_testMachineMemoryWindow);

  //  _testStandardCalls();
  if (m_nb_done_test==0)
    ARCANE_FATAL("No test done. Check environnment variable MESSAGE_PASSING_TEST");
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_testBarrier()
{
  IParallelMng* pm = m_parallel_mng;
  int nb_test = 3;
  info() << "Testing barrier (IParallelMng)";
  for( Integer i=0; i<nb_test; ++i )
    pm->barrier();

  IMessagePassingMng* mpm = pm->messagePassingMng();
  info() << "Testing barrier (IMessagePassingMng)";
  for( Integer i=0; i<nb_test; ++i )
    Arccore::MessagePassing::mpBarrier(mpm);

  IParallelNonBlockingCollective* pnbc = pm->nonBlockingCollective();
  if (pnbc){
    UniqueArray<Parallel::Request> requests;
    info() << "Testing NonBlockingBarrier";
    for( Integer i=0; i<nb_test; ++i ){
      requests.add(pnbc->barrier());
      requests.add(mpNonBlockingBarrier(mpm));
    }

    pm->waitAllRequests(requests);
  }
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_testSendRecvNonBlocking3()
{
  _testSendRecvNonBlockingSome(1,12,false);
  _testSendRecvNonBlockingSome(1,12,true);

  _testSendRecvNonBlockingSome(1,403239,false);
  _testSendRecvNonBlockingSome(1,234537,true);

  _testSendRecvNonBlockingSome(3,32239,false);
  _testSendRecvNonBlockingSome(3,78321,true);
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Meme test que le precedent mais on utilise WaitSome au lieu de WaitAll.

void ParallelMngTest::
_testSendRecvNonBlockingSome(Integer nb_message,Integer message_size,
                             bool is_non_blocking)
{
  IParallelMng* pm = m_parallel_mng;
  Int32 rank = pm->commRank();
  Int32 comm_size = pm->commSize();
  ITraceMng* tm = pm->traceMng();

  info() << "TestSendReceive: nb_message=" << nb_message
         << " message_size=" << message_size << " is_non_blocking=" << is_non_blocking;

  ValueChecker vc(A_FUNCINFO);
  Parallel::eWaitType wait_mode = (is_non_blocking) ? Parallel::WaitSomeNonBlocking : Parallel::WaitSome;
  if (rank==0){
    Int64 full_size = CheckedConvert::multiply(comm_size,nb_message,message_size);
    Int32UniqueArray all_mem_bufs(full_size);
    Array3View<Int32> all_bufs(all_mem_bufs.data(),comm_size,nb_message,message_size);
    Ref<IRequestList> requests = pm->createRequestListRef();
    const Integer nb_expected_request = nb_message * (comm_size-1);

    for( Integer orig=1; orig<comm_size; ++orig ){
      for( Integer z=0; z<nb_message; ++z ){
        requests->add(pm->recv(all_bufs[orig][z],orig,false));
      }
    }
    vc.areEqual(nb_expected_request,requests->size(),"Bad number of expected request");

    Integer received = 0;
    // Tableau pour vérifier que chaque requête a bien été appelée une et une seule fois.
    UniqueArray<Int32> nb_done_requests(nb_expected_request,0);
    UniqueArray<Int32> nb_expected_done_requests(nb_expected_request,1);
    Integer iteration = 0;
    Integer total_nb_done = 0;
    for( ;; ) {
      UniqueArray<Int32> ready;
      bool do_print = (iteration<50 || (iteration%100)==0);
      if (do_print)
        info() << "BEGIN WAIT iter=" << iteration << " nb_request=" << requests->size()
               << " wait_mode=" << wait_mode;
      Int32 nb_done = requests->wait(wait_mode);
      total_nb_done += nb_done;
      if (do_print)
        info() << "END WAIT iter=" << iteration << " nb_done=" << nb_done
               << " total=" << total_nb_done;

      if (nb_done==0){ // Plus de requêtes à attendre
        // En mode WaitSome, on sort uniquement s'il n'y a plus de requêtes
        // En node WaitSomeNonBlocking, on sort si on a fait autant de requêtes qu'attendu
        // car il est normal d'avoir \a nb_done==0 si on est en train d'attendre
        if (is_non_blocking){
          if (total_nb_done==nb_expected_request)
            break;
        }
        else
          break;
      }
      ++iteration;
      // Fait une petit pause de 1ms pour éviter une boucle trop rapide.
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      if (iteration>25000)
        ARCANE_FATAL("Too many iteration. probably a deadlock");
      // On récupère à partir du numéro de la requête le rang d'origine et
      // le numéro du message
      for( Integer iter_val : requests->doneRequestIndexes() ) {
        info() << "Receiving request request_id=" << iter_val;
        ++nb_done_requests[iter_val];
        Integer orig = iter_val/nb_message;
        Integer z = iter_val-orig*nb_message;
    		orig += 1; // On commence a 1.
    		received ++;
        info() << "Request orig=" << orig << " z=" << z << " first_value=" << all_bufs[orig][z][0];
    		if (m_verbose)
          for( Integer i=0; i<message_size; ++i ){
            tm->info() << "RECV orig=" << orig << " msg=" << z << " i=" << i << " v=" << all_bufs[orig][z][i];
          }
        for( Integer i=0; i<message_size; ++i ){
    			Int32 expected = orig*(nb_message+1) + z + i;
    			Int32 current  = all_bufs[orig][z][i];
    			if (current!=expected)
          	ARCANE_FATAL("Bad value expected={0} v={1} rank={2},{3},{4}",
                         expected,current,orig,z,i);
    		}
    	}
    }
		Int32 expected = nb_expected_request;
		Int32 current  = received;
    vc.areEqual(current,expected,"Bad number of reception expected");
    vc.areEqualArray(nb_done_requests,nb_expected_done_requests,"Bad number of wait for requests");
  }
  else{
    UniqueArray2<Int32> all_bufs(nb_message,message_size);
    UniqueArray<Request> requests;

    for( Integer z=0; z<nb_message; ++z ){
      for( Integer i=0; i<message_size; ++i ){
        all_bufs[z][i] = rank*(nb_message+1) + z + i;
      }
      tm->info() << "SEND orig=" << rank << " z=" << z << " first_value=" << all_bufs[z][0];
      if (m_verbose)
        for( Integer i=0; i<message_size; ++i ){
          tm->info() << "SEND orig=" << rank << " msg=" << z << " i=" << i << " v=" << all_bufs[z][i];
        }
      requests.add(pm->send(all_bufs[z],0,false));
    }

    pm->waitAllRequests(requests);
  }

  pm->barrier();

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_doTestSerializeSize(bool is_non_blocking)
{
  // L'implémentation MPI n'utilise pas le même mécanisme suivant la taille
  // du buffer. Il faut donc être sur de choisir les tailles qui testent les
  // deux mécanismes. Actuellement, la taille limite est de 50000 octets.
  // ATTENTION: l'argument passé à nb_value n'est pas la taille en octet
  // mais un nombre d'éléments.

  // Cette taille correspond à une mémoire utilisée d'environ 4Go par rang.
  // Il ne donc faire le test que si la machine a au moins une mémoire
  // supérieure à 4Go * commSize(). Cela permet de tester des messages
  // dont la taille est légement supérieure à 2Go (2^31).
  //_testSerializeSize(120000000);

  // Le test ci dessous permet de tester les messages de taille supérieur
  // à 4Go (2^32). Certaines implémentations MPI utilisent un 'unsigned int'
  // pour la taille de message et ce test permet de vérifier si ce n'est pas
  // le cas.
  //_testSerializeSize(220000000);

  if (is_non_blocking){
    _testNonBlockingSerializeSize(1000);
    _testNonBlockingSerializeSize(100017);
    _testNonBlockingSerializeSize(1500023);

    for( int i=0; i<2; ++i ){
      bool is_one_message = (i==1);

      _testProbeSerialize(1000,is_one_message);
      _testProbeSerialize(100017,is_one_message);
      _testProbeSerialize(1500023,is_one_message);
    }
  }
  else{
    _testSerializeSize(1000);
    _testSerializeSize(100017);
    _testSerializeSize(1500023);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_doTestSerializeMessageList()
{
  // TODO: pour l'instant on ne teste pas en séquentiel car cela n'est
  // pas implémenté mais il faudrait pouvoir utiliser le gestionnaire
  // en mémoire partagé pour gérer cela.
  if (!m_parallel_mng->isParallel())
    return;
  _doTestSerializeMessageList2(WaitAll);
  _doTestSerializeMessageList2(WaitSome);
  _doTestSerializeMessageList2(TestSome);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_doTestSerializeMessageList2(eWaitType wait_mode)
{
  // Comme pour _doTestSerializeSize(), il faudrait tester des messages
  // de plus de 4Go.

  _testSerializeMessageList(1000,wait_mode);
  _testSerializeMessageList(100017,wait_mode);
  _testSerializeMessageList(1500023,wait_mode);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_testSerialize1()
{
  info() << "Test _testSerialize1()";

  _doTestSerializeSize(true);
  _doTestSerializeSize(false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_doTestSerializerWithMessageInfo(bool use_wait_all)
{
  if (use_wait_all){
    _testSerializerWithMessageInfo(1049,true);
    _testSerializerWithMessageInfo(123829,true);
    _testSerializerWithMessageInfo(4093282,true);
  }
  else{
    _testSerializerWithMessageInfo(1049,false);
    _testSerializerWithMessageInfo(123829,false);
    _testSerializerWithMessageInfo(4093282,false);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_testSerializerWithMessageInfo()
{
  info() << "Test _testSerializerWithMessageInfo() wait_some";
  _doTestSerializerWithMessageInfo(false);
  info() << "Test _testSerializerWithMessageInfo() wait_all";
  _doTestSerializerWithMessageInfo(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParallelMngTest::SerializerTestValues
{
 public:
  void init(Integer nb_value)
  {
    // Création des tableaux contenant les valeurs de référence.
    ref_i16_values.resize(nb_value);
    for( Integer i=0, s=ref_i16_values.size(); i<s; ++i )
      ref_i16_values[i] = (Int16)(i+1);

    ref_i32_values.resize(nb_value);
    for( Integer i=0, s=ref_i32_values.size(); i<s; ++i )
      ref_i32_values[i] = i+1;

    ref_i64_values.resize(nb_value+10);
    for( Integer i=0, s=ref_i64_values.size(); i<s; ++i )
      ref_i64_values[i] = i+5;

    ref_real_values.resize(nb_value+3);
    for( Integer i=0, s=ref_real_values.size(); i<s; ++i )
      ref_real_values[i] = (Real)(i+27);

    ref_byte_values.resize(nb_value+8);
    for( Integer i=0, s=ref_byte_values.size(); i<s; ++i )
      ref_byte_values[i] = (Byte)( (i+23) % 255 );
  }

  void putValue(ISerializer* sb)
  {
    sb->setMode(ISerializer::ModeReserve);
    sb->reserveByte(ref_byte_values.size());
    sb->reserveInt16(ref_i16_values.size());
    sb->reserveInt32(ref_i32_values.size());
    sb->reserveInt64(ref_i64_values.size());
    sb->reserve(eBasicDataType::Real,ref_real_values.size());
    sb->allocateBuffer();
    sb->setMode(ISerializer::ModePut);
    sb->put(ref_byte_values);
    sb->put(ref_i16_values);
    sb->put(ref_i32_values);
    sb->put(ref_i64_values);
    sb->put(ref_real_values);
  }

  void getAndCheckValues(ISerializer* sb,ValueChecker& vc,const String& message)
  {
    sb->setMode(ISerializer::ModeGet);
    UniqueArray<Byte> byte_values(ref_byte_values.size());
    sb->get(byte_values);
    vc.areEqualArray(byte_values,ref_byte_values,message + " Byte");

    UniqueArray<Int16> i16_values(ref_i16_values.size());
    sb->get(i16_values);
    vc.areEqualArray(i16_values.span(),ref_i16_values.span(),message + " Int16");

    UniqueArray<Int32> i32_values(ref_i32_values.size());
    sb->get(i32_values);
    vc.areEqualArray(i32_values.span(),ref_i32_values.span(),message + " Int32");

    UniqueArray<Int64> i64_values(ref_i64_values.size());
    sb->get(i64_values);
    vc.areEqualArray(i64_values.span(),ref_i64_values.span(),message + " Int64");

    UniqueArray<Real> real_values(ref_real_values.size());
    sb->get(real_values);
    vc.areEqualArray(real_values.span(),ref_real_values.span(),message + " Real");
  }

 public:
  Int16UniqueArray ref_i16_values;
  Int32UniqueArray ref_i32_values;
  Int64UniqueArray ref_i64_values;
  RealUniqueArray ref_real_values;
  ByteUniqueArray ref_byte_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_testSerializeSize(Integer nb_value)
{
  IParallelMng* pm = m_parallel_mng;
  Int32 nb_rank = pm->commSize();
  Int32 my_rank = pm->commRank();
  Int32 min_rank = 0;
  ITraceMng* tm = pm->traceMng();
  info() << "Test serialize nb_value=" << nb_value
         << " test_broadcast=" << m_test_broadcast_serializer;

  // Création des tableaux contenant les valeurs de référence.
  SerializerTestValues test_values;
  test_values.init(nb_value);
  ValueChecker vc(A_FUNCINFO);

  // Certaines implémentations MPI ne supportent pas les collectives
  // dont la taille dépasse 2Go.
  // TODO: mettre INT64_MAX si le IParallelMng est en mémoire partagé
  // et INT32_MAX si on utilise MPI.

  constexpr Int64 broadcast_max_size = INT32_MAX;

  if (my_rank==0){
    SerializeBuffer sb;
    test_values.putValue(&sb);
    for( Integer i=min_rank; i<nb_rank; ++i ){
      if (i!=0){
        info() << "Send Serializer rank=" << i;
        pm->sendSerializer(&sb,i);
      }
    }
    if (m_test_broadcast_serializer){
      SerializeBuffer* buf = &sb;
      Int64 buf_size = buf->globalBuffer().size();
      info() << "Broadcast size=" << buf_size;
      pm->broadcast(Int64ArrayView(1,&buf_size),0);
      if (buf_size<broadcast_max_size){
        info() << "Master broadcast serializer";
        pm->broadcastSerializer(buf,0);
      }
      else
        warning() << "Do not broadcast big message size=" << buf_size;
    }
  }
  else{
    if (my_rank>=min_rank){
      SerializeBuffer sb;
      pm->recvSerializer(&sb,0);
      test_values.getAndCheckValues(&sb,vc,"Deserialize");
    }
    if (m_test_broadcast_serializer){
      SerializeBuffer sb;
      Int64 total_size = 0;
      pm->broadcast(Int64ArrayView(1,&total_size),0);
      info() << "Receive broadcasted size=" << total_size;
      if (total_size<broadcast_max_size){
        pm->broadcastSerializer(&sb,0);
      test_values.getAndCheckValues(&sb,vc,"Broadcast deserialize");
      }
    }
  }
  pm->barrier();
  tm->info() << " END TEST";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_testNonBlockingSerializeSize(Integer nb_value)
{
  IParallelMng* pm = m_parallel_mng;
  IMessagePassingMng* mpm = pm->messagePassingMng();
  Int32 nb_rank = pm->commSize();
  Int32 my_rank = pm->commRank();
  Int32 min_rank = 0;
  ITraceMng* tm = pm->traceMng();
  info() << "Test non blocking serialize nb_value=" << nb_value;

  // Création des tableaux contenant les valeurs de référence.
  SerializerTestValues test_values;
  test_values.init(nb_value);
  ValueChecker vc(A_FUNCINFO);

  Integer nb_message = 3;
  UniqueArray<Ref<ISerializeMessage>> requests;
  if (my_rank==0){
    for( Integer k=0; k<nb_message; ++k ){
      for( Integer i=min_rank; i<nb_rank; ++i ){
        if (i!=0){
          info() << "Send Serializer rank=" << i;
          auto x = mpCreateSendSerializeMessage(mpm, MessagePassing::MessageRank(i));
          test_values.putValue(x->serializer());
          requests.add(x);
        }
      }
    }
  }
  else{
    for( Integer k=0; k<nb_message; ++k ){
      if (my_rank>=min_rank){
        auto x = mpCreateReceiveSerializeMessage(mpm, MessagePassing::MessageRank(0));
        requests.add(x);
      }
    }
  }
  pm->processMessages(requests);
  if (my_rank!=0){
    for (Ref<ISerializeMessage>& s : requests)
      test_values.getAndCheckValues(s->serializer(),vc,"Deserialize");
  }
  pm->barrier();
  requests.clear();
  tm->info() << " END TEST";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_testProbeSerialize(Integer nb_value,bool use_one_message)
{
  IParallelMng* pm = m_parallel_mng;
  Int32 nb_rank = pm->commSize();
  Int32 my_rank = pm->commRank();
  Int32 min_rank = 0;
  ITraceMng* tm = pm->traceMng();
  info() << "Test probe serialize nb_value=" << nb_value
         << " is_one_message=" << use_one_message;
  tm->flush();

  // Création des tableaux contenant les valeurs de référence.
  SerializerTestValues test_values;
  test_values.init(nb_value);
  ValueChecker vc(A_FUNCINFO);

  Integer nb_message = 3;
  UniqueArray<ISerializeMessage*> requests;
  if (my_rank==0){
    for( Integer k=0; k<nb_message; ++k ){
      for( Integer i=min_rank; i<nb_rank; ++i ){
        if (i!=0){
          info() << "Send Serializer rank=" << i;
          auto x = pm->createSendSerializer(i);
          if (use_one_message)
            x->setStrategy(ISerializeMessage::eStrategy::OneMessage);
          test_values.putValue(x->serializer());
          requests.add(x);
        }
      }
    }
    pm->processMessages(requests);
  }
  else if (my_rank>min_rank){
    Integer nb_remaining_message = nb_message;
    while(nb_remaining_message>0){
      MessageTag mtag(MessagePassing::internal::BasicSerializeMessage::DEFAULT_SERIALIZE_TAG_VALUE);
      PointToPointMessageInfo p2p_info(MessageRank(0),mtag);
      p2p_info.setBlocking(false);
      MessageId id = pm->probe(p2p_info);
      if (id.isValid()){
        info() << "Recv probe nb_remaining=" << nb_remaining_message
               << " source_info_rank=" << id.sourceInfo().rank()
               << " source_info_tag=" << id.sourceInfo().tag();
        ScopedPtrT<ISerializeMessage> r(new SerializeMessage(my_rank,id));
        if (use_one_message)
          r->setStrategy(ISerializeMessage::eStrategy::OneMessage);
        requests.add(r.get());
        pm->processMessages(requests);
        test_values.getAndCheckValues(r->serializer(),vc,"Deserialize");
        --nb_remaining_message;
        requests.clear();
      }
    }
  }
  pm->barrier();
  for( auto& r : requests )
    delete r;
  tm->info() << " END TEST";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_testSerializeMessageList(Integer nb_value,eWaitType wait_mode)
{
  IParallelMng* pm = m_parallel_mng;
  Int32 nb_rank = pm->commSize();
  MessageRank my_rank(pm->commRank());
  Int32 min_rank = 0;
  ITraceMng* tm = pm->traceMng();
  info() << "Test SerializeMessageList nb_value=" << nb_value
         << " wait_mode=" << wait_mode;

  // Création des tableaux contenant les valeurs de référence.
  SerializerTestValues test_values;
  test_values.init(nb_value);
  ValueChecker vc(A_FUNCINFO);
  Ref<ISerializeMessageList> message_mng(pm->createSerializeMessageListRef());
  UniqueArray<Ref<ISerializeMessage>> messages;
  Integer nb_message = 3;
  //UniqueArray<ISerializeMessage*> requests;
  MessageRank sender_rank(0);
  if (my_rank==sender_rank){
    for( Integer k=0; k<nb_message; ++k ){
      for( Integer i=min_rank; i<nb_rank; ++i ){
        if (i!=0){
          info() << "Send Serializer rank=" << i;
          auto x = message_mng->createAndAddMessage(MessageRank(i),Parallel::MsgSend);
          test_values.putValue(x->serializer());
          messages.add(x);
        }
      }
    }
  }
  else{
    for( Integer k=0; k<nb_message; ++k ){
      if (my_rank.value()>=min_rank){
        auto x = message_mng->createAndAddMessage(sender_rank,Parallel::MsgReceive);
        messages.add(x);
      }
    }
  }
  if (wait_mode==WaitAll){
    message_mng->waitMessages(WaitAll);
    if (my_rank!=sender_rank){
      for( Ref<ISerializeMessage> s : messages ){
        // Il faut que le message soit terminé car on a fait un WaitAll.
        if (!s->finished())
          ARCANE_FATAL("Message is not finished");
        test_values.getAndCheckValues(s->serializer(),vc,"Deserialize");
      }
    }
  }
  else{
    UniqueArray<Ref<ISerializeMessage>> remaining_messages(messages);
    while (!remaining_messages.empty()){
      message_mng->waitMessages(wait_mode);
      messages = remaining_messages;
      remaining_messages.clear();
      for( Ref<ISerializeMessage> s : messages ){
        // Il faut que le message soit terminé car on a fait un WaitAll.
        if (s->finished()){
          if (my_rank!=sender_rank){
            test_values.getAndCheckValues(s->serializer(),vc,"Deserialize");
          }
        }
        else
          remaining_messages.add(s);
      }
    }
  }
  pm->barrier();
  tm->info() << " END TEST";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_testSerializerWithMessageInfo(Integer nb_value,bool use_wait)
{
  IParallelMng* pm = m_parallel_mng;
  Int32 nb_rank = pm->commSize();
  Int32 my_rank = pm->commRank();
  Int32 min_rank = 0;
  ITraceMng* tm = pm->traceMng();
  info() << "Test SerializerWithMessageInfo nb_value=" << nb_value
         << " use_wait=" << use_wait;

  // Création des tableaux contenant les valeurs de référence.
  SerializerTestValues test_values;
  test_values.init(nb_value);
  ValueChecker vc(A_FUNCINFO);

  Integer nb_message = 3;
  UniqueArray<ISerializer*> serializers;
  UniqueArray<Request> requests;
  // TODO: pouvoir changer le nombre de messages
  if (my_rank==0){
    for( Integer k=0; k<nb_message; ++k ){
      for( Integer i=min_rank; i<nb_rank; ++i ){
        if (i!=0){
          info() << "Send Serializer rank=" << i;
          auto x = new SerializeBuffer();
          serializers.add(x);
          test_values.putValue(x);
          Request r = pm->sendSerializer(x,{ MessageRank(i), Parallel::NonBlocking });
          requests.add(r);
        }
      }
    }
  }
  else{
    for( Integer k=0; k<nb_message; ++k ){
      if (my_rank>=min_rank){
        info() << "Receive Serializer rank=" << my_rank;
        auto x = new SerializeBuffer();
        serializers.add(x);
        Request r = pm->receiveSerializer(x,{ MessageRank(0), Parallel::NonBlocking });
        requests.add(r);
      }
    }
  }
  // TODO: ajouter test avec IRequestList et les trois modes de wait.
  if (use_wait){
    info() << "WaitAll requests";
    pm->waitAllRequests(requests);
  }
  else{
    info() << "WaitSome requests";
    UniqueArray<Request> requests2(requests);
    while (!requests2.empty()){
      UniqueArray<Integer> done_indexes = pm->waitSomeRequests(requests2);
      UniqueArray<bool> is_done_request(requests2.size(),false);
      for( Integer x : done_indexes ){
        info() << "IS_DONE idx=" << x;
        is_done_request[x] = true;
      }
      requests.clear();
      for( Integer i=0, n=is_done_request.size(); i<n; ++i ){
        if (!is_done_request[i])
          requests.add(requests2[i]);
      }
      requests2 = requests;
    }
  }
  if (my_rank!=0){
    for( ISerializer* s : serializers )
      test_values.getAndCheckValues(s,vc,"Deserialize");
  }
  pm->barrier();
  for( ISerializer* s : serializers )
    delete s;
  tm->info() << " END TEST";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void ParallelMngTest::
_testParallelBasic(DataType data)
{
  IParallelMng* pm = m_parallel_mng;
  DataType data2 = pm->reduce(Parallel::ReduceSum,data);
  DataType data3 = pm->reduce(Parallel::ReduceMax,data);
  DataType data4 = pm->reduce(Parallel::ReduceMin,data);
  info() << "** DATA2_SUM=" << data2 << " data_max=" << data3 << " data_min=" << data4;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_testReduce2()
{
  Int32 sid = m_parallel_mng->commRank();
  Real v0 = (Real)(1+sid);
  _testParallelBasic(Real(v0));
  _testParallelBasic(Real2(v0,v0+1.0));
  _testParallelBasic(Real2x2::fromLines(v0,v0+1.0,v0+2.0,v0+3.0));
  _testParallelBasic(Real3(v0,v0+1.0,v0+2.0));
  _testParallelBasic(Real3x3::fromLines(v0,v0+1.0,v0+2.0,v0+3.0,v0+4.0,v0+5.0,v0+6.0,v0+7.0,v0+8.0));
  _testParallelBasic(HPReal(math::log(3.0),math::log(3.14159)));
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_testProcessMessages()
{
  info() << "Test: TestProcessMessage";
  _testProcessMessages(nullptr);
  {
    ParallelExchangerOptions options;
    info() << "Test: TestProcessMessage with collective";
    options.setExchangeMode(ParallelExchangerOptions::EM_Collective);
    _testProcessMessages(&options);
  }
 {
    ParallelExchangerOptions options;
    info() << "Test: TestProcessMessage with max pending";
    options.setMaxPendingMessage(5);
    _testProcessMessages(&options);
  }

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_testProcessMessages(const ParallelExchangerOptions* exchange_options)
{
  IParallelMng* pm = m_parallel_mng;
  Int32 rank = pm->commRank();
  Int32 nb_rank = pm->commSize();
  ITraceMng* tm = pm->traceMng();

  auto exchanger { ParallelMngUtils::createExchangerRef(pm) };
  exchanger->setVerbosityLevel(2);
  exchanger->setName("TestProcessMessage");

  Int32 nb_send = nb_rank;
  for( Int32 i=0; i<nb_send; ++i ){
    exchanger->addSender(i);
  }
  exchanger->initializeCommunicationsMessages();
  Integer base_size = 32;
  for( Int32 i=0; i<nb_send; ++i ){
    ISerializeMessage* sm = exchanger->messageToSend(i);
    ISerializer* s = sm->serializer();
    Int32 dest_rank = sm->destination().value();
    Integer message_size = base_size + dest_rank + rank;
    s->setMode(ISerializer::ModeReserve);
    s->reserveInteger(1); // Pour le nombre d'elements
    s->reserveInt32(message_size); // Pour les elements
    s->allocateBuffer();
    s->setMode(ISerializer::ModePut);
    s->putInteger(message_size);
    Int32UniqueArray msg(message_size);
    for( Integer z=0; z<message_size; ++z ){
      msg[z] = rank + z + i;
    }
    s->put(msg);
  }
  if (exchange_options)
    exchanger->processExchange(*exchange_options);
  else
    exchanger->processExchange();
  tm->info() << "END EXCHANGE";
  {
    Integer nb_receiver = exchanger->nbReceiver();
    tm->info() << "NB RECEIVER=" << nb_receiver;
    Int32UniqueArray received_msg;
    for( Integer i=0; i<nb_receiver; ++i ){
      ISerializeMessage* sm = exchanger->messageToReceive(i);
      Int32 orig_rank = sm->destination().value();
      ISerializer* s = sm->serializer();
      s->setMode(ISerializer::ModeGet);
      Integer nb_info = s->getInteger();
      Integer expected_nb_info = base_size + orig_rank + rank;

      if (nb_info!=expected_nb_info)
        ARCANE_FATAL("Bad message size v={0} expected={1} orig_rank={2} my_rank={3}",
                     nb_info,expected_nb_info,orig_rank,rank);

      //info() << "RECEIVE NB_INFO=" << nb_info << " from=" << orig_rank;
      received_msg.resize(nb_info);
      s->get(received_msg);
      for( Integer z=0; z<nb_info; ++z ){
        Int32 current = received_msg[z];
        Int32 expected = orig_rank + rank + z;
        if (current!=expected)
          ARCANE_FATAL("Bad compare value v={0} expected={1} orig_rank={2} index={3} my_rank={4}",
                       current,expected,orig_rank,z,rank);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_testMachineMemoryWindow()
{
  {
    // nb_elem doit être paire pour ce test.
    //![snippet_arcanedoc_parallel_shmem_usage_1]
    constexpr Integer nb_elem = 14;

    IParallelMng* pm = m_parallel_mng;
    Integer my_rank = pm->commRank();

    MachineMemoryWindow<Integer> window(pm, nb_elem);
    //![snippet_arcanedoc_parallel_shmem_usage_1]

    //![snippet_arcanedoc_parallel_shmem_usage_2]
    ConstArrayView<Int32> machine_ranks(window.machineRanks());
    Integer machine_nb_proc = machine_ranks.size();
    //![snippet_arcanedoc_parallel_shmem_usage_2]

    {
      Ref<IParallelTopology> topo = ParallelMngUtils::createTopologyRef(pm);
      if (topo->machineRanks().size() != machine_ranks.size()) {
        // Problème avec MPI. Peut intervenir si MPICH est compilé en mode ch3:sock.
        // On ne plante pas les tests dans ce cas.
        warning() << "Shared memory not supported"
                  << " -- Nb machine ranks with ParallelTopo : " << topo->machineRanks().size()
                  << " -- Nb machine ranks with MPI_COMM_TYPE_SHARED : " << machine_ranks.size();
        return;
      }
    }
    if (window.windowConstView().size() != machine_nb_proc * nb_elem) {
      ARCANE_FATAL("Bad sizeWindow()");
    }

    //![snippet_arcanedoc_parallel_shmem_usage_3]
    {
      Span av_my_segment(window.segmentView());

      Integer iter = 0;
      for (Integer& elem : av_my_segment) {
        elem = iter * (my_rank + 1);
        iter++;
      }
    }
    window.barrier();
    //![snippet_arcanedoc_parallel_shmem_usage_3]

    for (Int32 rank : machine_ranks) {
      Span av_segment(window.segmentView(rank));

      for (Integer i = 0; i < nb_elem; ++i) {
        //info() << "Test " << i << " : " << av_segment[i] << " -- " << rank;
        if (av_segment[i] != i * (rank + 1)) {
          ARCANE_FATAL("Bad element in memory window -- Expected : {0} -- Found : {1}", (i * (rank + 1)), av_segment[i]);
        }
      }
    }

    //![snippet_arcanedoc_parallel_shmem_usage_4]
    for (Int32 rank : machine_ranks) {
      Span av_segment(window.segmentConstView(rank));

      for (Integer i = 0; i < nb_elem; ++i) {
        if (av_segment[i] != i * (rank + 1)) {
          ARCANE_FATAL("Bad element in memory window -- Expected : {0} -- Found : {1}", (i * (rank + 1)), av_segment[i]);
        }
      }
    }
    //![snippet_arcanedoc_parallel_shmem_usage_4]

    //![snippet_arcanedoc_parallel_shmem_usage_5]
    window.barrier();

    constexpr Integer nb_elem_div = nb_elem / 2;

    window.resizeSegment(nb_elem_div);
    //![snippet_arcanedoc_parallel_shmem_usage_5]

    for (Int32 rank : machine_ranks) {
      Span av_segment(window.segmentConstView(rank));

      for (Integer i = 0; i < nb_elem_div; ++i) {
        //info() << "Test2 " << i << " : " << av_segment[i] << " -- " << rank;
        Int32 procdiv2 = rank / 2;
        Integer i2 = (rank % 2 == 0 ? i : i + nb_elem_div);
        if (av_segment[i] != i2 * (procdiv2 + 1)) {
          ARCANE_FATAL("Bad element in memory window -- Expected : {0} -- Found : {1}", (i * (rank + 1)), av_segment[i]);
        }
      }
    }

    window.barrier();
    window.resizeSegment(nb_elem);

    //![snippet_arcanedoc_parallel_shmem_usage_6]
    if (my_rank == machine_ranks[0]) {
      Span av_window(window.windowView());
      for (Integer j = 0; j < machine_nb_proc; ++j) {
        for (Integer i = 0; i < nb_elem; ++i) {
          av_window[i + (j * nb_elem)] = machine_ranks[j];
        }
      }
    }
    window.barrier();

    {
      Span av_window(window.windowConstView());
      for (Integer j = 0; j < machine_nb_proc; ++j) {
        for (Integer i = 0; i < nb_elem; ++i) {
          if (av_window[i + (j * nb_elem)] != machine_ranks[j]) {
            ARCANE_FATAL("Bad element in memory window -- Expected : {0} -- Found : {1}", machine_ranks[j], av_window[i + (j * nb_elem)]);
          }
        }
      }
    }
    window.barrier();
    //![snippet_arcanedoc_parallel_shmem_usage_6]
  }

  {
    IParallelMng* pm = m_parallel_mng;
    Integer my_rank = pm->commRank();

    ArrayView<Integer> my_rank_av(1, &my_rank);
    DynamicMachineMemoryWindow<Integer> test(pm, 1);
    ConstArrayView machine_ranks(test.machineRanks());

    Int32 pos_in_machine_ranks = -1;
    for (Integer i = 0; i < machine_ranks.size(); ++i) {
      if (machine_ranks[i] == my_rank) {
        pos_in_machine_ranks = i;
        break;
      }
    }
    if (pos_in_machine_ranks == -1) {
      ARCANE_FATAL("Rank is not in machine -- my_rank : {0} -- ranks : {1}", my_rank, machine_ranks);
    }
    test.segmentView()[0] = my_rank;

    {
      Int32 add_in = -1;

      if (my_rank % 2 == 0 && pos_in_machine_ranks + 1 < machine_ranks.size()) {
        add_in = machine_ranks[pos_in_machine_ranks + 1];
      }
      if (my_rank % 2 == 1 && pos_in_machine_ranks - 1 >= 0) {
        add_in = machine_ranks[pos_in_machine_ranks - 1];
      }
      if (add_in == -1) {
        test.addToAnotherSegment();
      }
      else {
        test.addToAnotherSegment(add_in, my_rank_av);
      }
      if (my_rank % 2 == 0 && pos_in_machine_ranks + 1 < machine_ranks.size()) {
        ARCANE_ASSERT(machine_ranks[pos_in_machine_ranks + 1] == test.segmentConstView()[test.segmentConstView().size() - 1], ("Bad elem"));
      }
      if (my_rank % 2 == 1 && pos_in_machine_ranks - 1 >= 0) {
        ARCANE_ASSERT(machine_ranks[pos_in_machine_ranks - 1] == test.segmentConstView()[test.segmentConstView().size() - 1], ("Bad elem"));
      }
    }
    {
      Int32 add_in = -1;
      if (my_rank % 2 == 1 && pos_in_machine_ranks + 1 < machine_ranks.size()) {
        add_in = machine_ranks[pos_in_machine_ranks + 1];
      }
      if (my_rank % 2 == 0 && pos_in_machine_ranks - 1 >= 0) {
        add_in = machine_ranks[pos_in_machine_ranks - 1];
      }
      if (add_in == -1) {
        test.addToAnotherSegment();
      }
      else {
        test.addToAnotherSegment(add_in, my_rank_av);
      }
      if (my_rank % 2 == 1 && pos_in_machine_ranks + 1 < machine_ranks.size()) {
        ARCANE_ASSERT(machine_ranks[pos_in_machine_ranks + 1] == test.segmentConstView()[test.segmentConstView().size() - 1], ("Bad elem"));
      }
      if (my_rank % 2 == 0 && pos_in_machine_ranks - 1 >= 0) {
        ARCANE_ASSERT(machine_ranks[pos_in_machine_ranks - 1] == test.segmentConstView()[test.segmentConstView().size() - 1], ("Bad elem"));
      }
    }
    debug() << "Test : " << test.segmentConstView();
    test.resize(0);
    test.reserve(15);
    {
      UniqueArray<Integer> ref(15);
      for (Integer i = 0; i < 15; ++i) {
        Int32 add_in = (my_rank + i) % machine_ranks.size();
        test.addToAnotherSegment(add_in, my_rank_av);
        ref[i] = (((my_rank - i) % machine_ranks.size()) + machine_ranks.size()) % machine_ranks.size();
      }
      debug() << "Test : " << test.segmentConstView();
      debug() << "Ref : " << ref;
      ARCANE_ASSERT(ref == test.segmentConstView(), ("Result is not egal to ref"));
    }
    {
      Int32 add_in = -1;
      if (my_rank % 2 == 1 && pos_in_machine_ranks - 1 >= 0) {
        add_in = machine_ranks[pos_in_machine_ranks - 1];
      }

      if (add_in == -1) {
        test.addToAnotherSegment();
        test.resize();
        test.add();
      }
      else {
        test.addToAnotherSegment(add_in, test.segmentConstView());
        test.resize(0);
        test.add(test.segmentConstView(add_in).subSpan(0, 15)); // Ne fonctionne pas sans reserve.
      }
    }
    debug() << "Test : " << test.segmentConstView();
    test.shrink();
  }
  {
    //![snippet_arcanedoc_parallel_shmem_usage_7]
    IParallelMng* pm = m_parallel_mng;
    Integer my_rank = pm->commRank();
    DynamicMachineMemoryWindow<Integer> window(pm, 5);
    ConstArrayView machine_ranks(window.machineRanks());
    //![snippet_arcanedoc_parallel_shmem_usage_7]
    {
      //![snippet_arcanedoc_parallel_shmem_usage_8]
      DynamicMachineMemoryWindow<Integer> window2(pm);
      //![snippet_arcanedoc_parallel_shmem_usage_8]
    }
    //![snippet_arcanedoc_parallel_shmem_usage_9]
    {
      Span av_my_segment(window.segmentView());
      Integer iter = 0;
      for (Integer& elem : av_my_segment) {
        elem = iter * (my_rank + 1);
        iter++;
      }
    }
    window.barrier();
    for (Int32 rank : machine_ranks) {
      Span av_segment(window.segmentConstView(rank));
      for (Integer i = 0; i < 5; ++i) {
        if (av_segment[i] != i * (rank + 1)) {
          ARCANE_FATAL("Bad element in memory window -- Expected : {0} -- Found : {1}", (i * (rank + 1)), av_segment[i]);
        }
      }
    }
    //![snippet_arcanedoc_parallel_shmem_usage_9]
    //![snippet_arcanedoc_parallel_shmem_usage_10]
    Integer pos_in_machine_ranks = -1;
    for (Integer i = 0; i < machine_ranks.size(); ++i) {
      if (my_rank == machine_ranks[i]) {
        pos_in_machine_ranks = i;
        break;
      }
    }
    // Remarque : ici, pos_in_machine_ranks correspond au rang du processus
    // dans le communicateur MPI "machine".
    {
      UniqueArray<Integer> buf;
      if (pos_in_machine_ranks == 0) {
        for (Integer i = 0; i < 10; ++i) {
          buf.add(i);
        }
      }
      window.add(buf);
    }
    //![snippet_arcanedoc_parallel_shmem_usage_10]
    //![snippet_arcanedoc_parallel_shmem_usage_11]
    window.reserve(20);
    //![snippet_arcanedoc_parallel_shmem_usage_11]
    //![snippet_arcanedoc_parallel_shmem_usage_12]
    window.resize(12);
    info() << window.segmentConstView().size();
    //![snippet_arcanedoc_parallel_shmem_usage_12]
    //![snippet_arcanedoc_parallel_shmem_usage_13]
    Int32 voisin = -1;
    if (my_rank % 2 == 0 && pos_in_machine_ranks + 1 < machine_ranks.size()) {
      voisin = machine_ranks[pos_in_machine_ranks + 1];
    }
    else if (my_rank % 2 == 1 && pos_in_machine_ranks - 1 >= 0) {
      voisin = machine_ranks[pos_in_machine_ranks - 1];
    }

    // On efface les éléments déjà présents dans les segments.
    window.resize(0);

    // Si l'on n'a pas de voisins, on ajoute rien.
    if (voisin == -1) {
      window.addToAnotherSegment();
    }
    else {
      UniqueArray<Integer> buf;
      for (Integer i = 0; i < 10; ++i) {
        buf.add(my_rank);
      }
      window.addToAnotherSegment(voisin, buf);
    }
    info() << "Segment final : " << window.segmentConstView();
    window.shrink();
    //![snippet_arcanedoc_parallel_shmem_usage_13]
  }

  {
    IParallelMng* pm = m_parallel_mng;
    DynamicMachineMemoryWindowMemoryAllocator memory_allocator(pm);

    {
      UniqueArray<Integer> array(&memory_allocator, 10);

      for (Integer& a : array) {
        a = pm->commRank();
      }

      info() << "array : " << array;
    }

    {
      AllocatedMemoryInfo ptr_info = memory_allocator.allocate({}, sizeof(Integer));
      auto* ptr = static_cast<Integer*>(ptr_info.baseAddress());
      ptr[0] = pm->commRank();

      info() << "ptr[0] : " << ptr[0];

      memory_allocator.deallocate({}, ptr_info);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_testBroadcastSerializer()
{
  _testBroadcastSerializer2(500);
  _testBroadcastSerializer2(5);
  _testBroadcastSerializer2(3);
  _testBroadcastSerializer2(2);
  _testBroadcastSerializer2(25000);
  _testBroadcastSerializer2(123412);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_testBroadcastSerializer2(Integer n)
{
  Int32 rank = m_parallel_mng->commRank();
  Int32 master_rank = 0;

  SerializeBuffer buffer;
  buffer.setMode(ISerializer::ModeReserve);
  buffer.reserveInteger(2*n);
  buffer.reserve(DT_Real,6*n);
  buffer.allocateBuffer();

  if ( rank == master_rank ) {
    buffer.setMode(ISerializer::ModePut);
    for( Integer i = 0 ; i < n ; ++i ){
      buffer.putInteger(1);
      buffer.putInteger(2);
    }
    for( Integer i = 0; i<n ; ++i ){
      buffer.put(1.0);
      buffer.put(2.0);
      buffer.put(3.0);
      buffer.put(4.0);
      buffer.put(5.0);
      buffer.put(6.0);
    }
  }
  info() << "Avant broadcast buffer, n = " << n;
  m_parallel_mng->broadcastSerializer(&buffer, master_rank);
  info() << "Apres broadcast buffer, n = " << n;
  if (rank!=master_rank){
    buffer.setMode(ISerializer::ModeGet);
    for( Integer i = 0 ; i < n ; ++i ){
      for( Integer j = 0; j<2 ; ++j ){
        Int32 p = buffer.getInteger();
        if (p!=(j+1))
          ARCANE_FATAL("Bad compare Integer value v={0} expected={1} orig_rank={2} my_rank={3}",
                       p,j,master_rank,rank);
      }
    }
    for( Integer i = 0; i<n ; ++i ){
      for( Integer j = 0; j<6 ; ++j ){
        Real r = buffer.getReal();
        if (r!=(Real)(j+1))
          ARCANE_FATAL("Bad compare Real value v={0} expected={1} orig_rank={2} my_rank={3}",
                       r,(j+1),master_rank,rank);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Cette méthode est dans ParallelMngDataTypeTest
extern "C++" void
_testParallelMngDataType(IParallelMng* pm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_testStandardCalls()
{
  _testParallelMngDataType(m_parallel_mng);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_testTopology()
{
  auto pt { ParallelMngUtils::createTopologyRef(m_parallel_mng) };
  ITraceMng* tm = m_parallel_mng->traceMng();
  
  Int32ConstArrayView master_machine_ranks = pt->masterMachineRanks();
  for( Integer i=0, n=master_machine_ranks.size(); i<n; ++i )
    tm->info() << "Machine master rank=" << master_machine_ranks[i] << "/" << n;

  Int32ConstArrayView master_process_ranks = pt->masterProcessRanks();
  for( Integer i=0, n=master_process_ranks.size(); i<n; ++i )
    tm->info() << "Process master rank=" << master_process_ranks[i] << "/" << n;

  tm->info() << "Rank in Machine list=" << pt->machineRank();
  tm->info() << "Rank in Process list=" << pt->processRank();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_testBroadcastStringAndMemoryBuffer()
{
  String s1 = "Ceci est un test";
  _testBroadcastStringAndMemoryBuffer2(s1);
  String s2 = "Ceci est un ajout éàADùX";
  for( Integer i=0; i<10; ++i )
    s1 = s1 + s2;
  _testBroadcastStringAndMemoryBuffer2(s2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_testBroadcastStringAndMemoryBuffer2(const String& wanted_str)
{
  info() << "Testing broadcast string and memorybuffer";

  IParallelMng* pm = m_parallel_mng;
  Int32 my_rank = pm->commRank();
  
  ValueChecker vc(A_FUNCINFO);
  UniqueArray<Byte> ref_values(wanted_str.utf8());
  if (my_rank==0){
    String s2 = wanted_str;
    pm->broadcastString(s2,0);
    pm->broadcastMemoryBuffer(ref_values,0);
  }
  else{
    String s2;
    pm->broadcastString(s2,0);
    UniqueArray<Byte> recv_values;
    vc.areEqual(s2,wanted_str,"Bad broadcast string");
    pm->broadcastMemoryBuffer(recv_values,0);
    vc.areEqual(ref_values,recv_values,"Bad broadcast memory");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_testNamedBarrier()
{
  info() << "Testing named barrier";

  IParallelMng* pm = m_parallel_mng;
  Int32 my_rank = pm->commRank();
  {
    String barrier_name = "ThisIsMyBarrier";
    MessagePassing::namedBarrier(pm,barrier_name);
    info() << "First test ok for named barrier";
  }

  // Test avec une longue chaine de caractères
  {
    char buf[2048];
    for( Integer i=0; i<2000; ++i )
      buf[i] = (char)('a' + (i%32));
    buf[2000] = '\0';
    MessagePassing::namedBarrier(pm,String(buf));
    info() << "Test ok for long named barrier";
  }

  if (pm->isParallel()){
    String barrier_name = "ThisIsMyBarrier2";
    bool has_exception = false;
    if (my_rank==0){
      barrier_name = "ThisIsBarrier0";
    }
    try{
      MessagePassing::namedBarrier(pm,barrier_name);
    }
    catch(const FatalErrorException& ex){
      has_exception = true;
    }
    // Seul le rang 0 doit lever une exception (attention cela
    // dépendant des valeurs de barrier_name car l'implémentation
    // utilise une reduction max).
    if (my_rank==0)
      if (!has_exception)
        ARCANE_FATAL("No exception for named barrier for rank 0");
    if (my_rank!=0)
      if (has_exception)
        ARCANE_FATAL("Unexpected exception for named barrier for rank!=0");
    info() << "Test ok for named barrier with different name";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTest::
_launchTest(const String& test_name,void (ParallelMngTest::*func)())
{
  //info() << "CheckTest current_test=" << test_name << " valid=" << m_test_name;
  if (m_test_name=="all" || m_test_name==test_name){
    ITraceMng* tm = m_parallel_mng->traceMng();
    tm->info() << "Test " << test_name;
    (this->*func)();
    tm->info() << "Test " << test_name << " finished";
    tm->flush();
    ++m_nb_done_test;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParallelMngTestService
: public AbstractService
, public IDirectExecution
{
 public:

  ParallelMngTestService(const ServiceBuildInfo& sb)
  : AbstractService(sb){}

 public:

  void build() override {}

  //! Exécute l'opération du service
  void execute() override;

  //! Vrai si le service est actif
  bool isActive() const override { return true; }

  void setParallelMng(IParallelMng* pm) override
  {
    m_main_parallel_mng = pm;
  }

 private:

  IParallelMng* m_main_parallel_mng = nullptr;
  String m_test_name;

 private:

  void _doExecute(IParallelMng* pm);
  void _doExecuteSub(IParallelMng* pm);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTestService::
_doExecute(IParallelMng* pm)
{
  info() << "** ** ** EXECUTE TEST nb_rank=" << pm->commSize();
  ParallelMngTest tester(pm,m_test_name);
  tester.execute();
}

void ParallelMngTestService::
_doExecuteSub(IParallelMng* pm)
{
  info() << "DO SUB_PARALLEL_MNG";
  Int32 nb_rank = pm->commSize();
  // TODO: le plus simple serait de tester cela de manière récursive en
  // gardant à chaque fois 1 rang sur 2.
  // On pourra aussi à terme tester des choses plus compliquées comme ne
  // pas garder le rang 0 et/ou pas le même nombre de rangs locaux en
  // mode hybride (mais cela n'est pas supporté pour le moment).

  // Les tests avec 4 PE sont en mode MPI ou mémoire partagée
  if (nb_rank==4){
    UniqueArray<Int32> kept_ranks;
    // Prend 1 coeur sur 2.
    for( Integer i=0; i<nb_rank; ++i )
      if ((i%2)==0)
        kept_ranks.add(i);
    Ref<IParallelMng> sub_pm = pm->createSubParallelMngRef(kept_ranks);
    if (sub_pm.get())
      _doExecute(sub_pm.get());
  }
  // Les tests avec 12 PE sont en mode hybride (3 MPI * 4 threads)
  if (nb_rank==12){
    // En mode hybride, chaque processus MPI doit avoir le même nombre
    // de rang en mémoire partagée. On teste 3 MPI * 2 threads (1 coeur sur 2)
    // et 3 MPI uniquement (1 coeur sur 4) ce qui revient à faire comme
    // si on était en mode MPI pure.

    UniqueArray<Int32> kept_ranks;

    const bool do_one = false;
    if (do_one){
      // Prend un rang sur 2
      for( Integer i=0; i<nb_rank; ++i )
        if ((i%2)==0)
          kept_ranks.add(i);
      {
        Ref<IParallelMng> sub_pm = pm->createSubParallelMngRef(kept_ranks);
        if (sub_pm.get())
          _doExecute(sub_pm.get());
      }
    }

    bool do_4 = true;
    if (do_4){
      // Prend un rang sur 4
      kept_ranks.clear();
      for( Integer i=0; i<nb_rank; ++i )
        if ((i%4)==0)
          kept_ranks.add(i);
      {
        Ref<IParallelMng> sub_pm = pm->createSubParallelMngRef(kept_ranks);
        if (sub_pm.get())
          _doExecute(sub_pm.get());
      }
    }
  }

  // Teste le sous-communicateur à la MPI_Comm_split.
  // Ne fonctionne qu'avec MPI.
  if (((nb_rank % 2) == 0) && !pm->isThreadImplementation() && !pm->isHybridImplementation()) {
    info() << "Test SubParallelMng with (color,key) nb_rank=" << nb_rank;
    Int32 my_rank = pm->commRank();
    Int32 middle = nb_rank / 2;
    // Créé deux instances. Une avec les (nb_rank/2) premiers rangs et une avec les autres.
    Int32 color = 1;
    Int32 expected_total = (middle * (middle + 1)) / 2;
    if (my_rank >= middle) {
      color = 2;
      expected_total = ((nb_rank * (nb_rank + 1)) / 2) - expected_total;
    }
    Ref<IParallelMng> sub_pm = ParallelMngUtils::createSubParallelMngRef(pm, color, my_rank);
    ARCANE_CHECK_POINTER(sub_pm.get());
    // Pour vérifier que tout est Ok, on fait une réduction avec comme valeur
    // notre (rang+1) et on doit trouver la somme de N entiers consécutifs.
    Int32 total = sub_pm->reduce(ReduceSum, my_rank + 1);
    Int32 sub_nb_rank = sub_pm->commSize();
    if (sub_nb_rank != middle)
      ARCANE_FATAL("Bad number of rank n={0} expected={1}", sub_nb_rank, middle);
    info() << "Total=" << total << " expected=" << expected_total;
    if (total != expected_total)
      ARCANE_FATAL("Bad value total={0} expected={1}", total, expected_total);
  }
}  

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngTestService::
execute()
{
  m_test_name = platform::getEnvironmentVariable("MESSAGE_PASSING_TEST");

  // Si le nom du test commence par 'sub', cela signifie qu'il faut créer
  // et utiliser les sous IParallelMng.
  bool do_sub = false;
  if (m_test_name.startsWith("sub_")){
    do_sub = true;
    m_test_name = m_test_name.substring(4);
  }

  ARCANE_CHECK_POINTER(m_main_parallel_mng);
  IParallelMng* pm = m_main_parallel_mng;

  if (do_sub){
    _doExecuteSub(pm);
  }
  else
    _doExecute(pm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_APPLICATION_FACTORY(ParallelMngTestService,
                                    IDirectExecution,ParallelMngTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Test exécution des coeurs non alloués pour les sous-domaines.
 */
class ParallelTestIdleService
: public AbstractService
, public IDirectExecution
{
 public:

  ParallelTestIdleService(const ServiceBuildInfo& sb)
  : AbstractService(sb), m_parallel_mng(nullptr){}

  void build() override {}

 public:

  //! Exécute l'opération du service
  void execute() override
  {
    info() << "TEST ParallelTestIdleService";
  }

  //! Vrai si le service est actif
  bool isActive() const override { return true; }

  void setParallelMng(IParallelMng* pm) override
  {
    m_parallel_mng = pm;
  }

 private:

  IParallelMng* m_parallel_mng;

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_APPLICATION_FACTORY(ParallelTestIdleService,
                                    IDirectExecution,ParallelTestIdleService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
