cmake_minimum_required(VERSION 3.8)

aux_source_directory(. SRC)

include_directories("../")

add_library(sslib ${SRC})
target_link_libraries(sslib ${OPENSSL_LIBRARIES} pthread dl)
#target_link_libraries(sslib openssl_crypto openssl_ssl pthread)
#target_link_libraries(sslib OpenSSL::Crypto OpenSSL::SSH pthread)
#if(${enable_static_linking})
#	target_link_libraries(sslib OpenSSL::Crypto OpenSSL::SSH pthread)
#else()
#	target_link_libraries(sslib crypto pthread)
#endif()
