/*===-- serialbox-c/Serializer.h ----------------------------------------------------*- C++ -*-===*\
 *
 *                                    S E R I A L B O X
 *
 * This file is distributed under terms of BSD license.
 * See LICENSE.txt for more information
 *
 *===------------------------------------------------------------------------------------------===//
 *
 *! \file
 *! This file contains the C implementation of the Serializer.
 *
\*===------------------------------------------------------------------------------------------===*/

#ifndef SERIALBOX_C_SERIALIZER_H
#define SERIALBOX_C_SERIALIZER_H

#include "serialbox-c/Type.h"

#ifdef __cplusplus
extern "C" {
#endif

/*===------------------------------------------------------------------------------------------===*\
 *     Construction & Destruction
\*===------------------------------------------------------------------------------------------===*/

/**
 * \brief Create a new Serializer
 *
 * \param mode         Mode of the Serializer
 * \param directory    Directory of the Archive and Serializer meta-data
 * \param prefix       Prefix of all filenames
 * \param archiveName  Name of Archive (e.g "BinaryArchive")
 *
 * \return refrence to the newly constructed Serializer or NULL if an error occured
 *
 * This will read ´MetaData-prefix.json´ to initialize the Serializer and construct the Archive by
 * reading the ´ArchiveMetaData-prefix.json´.
 */
serialboxSerializer_t serialboxSerializerCreate(serialboxOpenModeKind mode, const char* directory,
                                                const char* prefix, const char* archive);

/**
 * \brief Destroy the serializer and deallocate all memory
 *
 * \param serializerPtr  Pointer to Serializer to use
 */
void serialboxSerializerDestroy(serialboxSerializer_t* serializerPtr);

/*===------------------------------------------------------------------------------------------===*\
 *     Utility
\*===------------------------------------------------------------------------------------------===*/

/**
 * \brief Return mode of the Serializer
 *
 * \param serializer  Serializer to use
 * \return mode of the Serializer
 */
serialboxOpenModeKind serialboxSerializerGetMode(const serialboxSerializer_t serializer);

/**
 * \brief Return the directory of the Serializer
 *
 * \param serializer  Serializer to use
 * \return directory of the Serializer as a null-terminated string
 */
const char* serialboxSerializerGetDirectory(const serialboxSerializer_t serializer);

/**
 * \brief Return the prefix of all filenames
 *
 * \param serializer  Serializer to use
 * \return prefix of the Serializer as a null-terminated string
 */
const char* serialboxSerializerGetPrefix(const serialboxSerializer_t serializer);

/**
 * \brief Write meta-data to disk
 *
 * \param serializer  Serializer to use
 */
void serialboxSerializerUpdateMetaData(serialboxSerializer_t serializer);

/**
 * \brief Indicate whether serialization is enabled [default: enabled]
 */
extern int serialboxSerializationEnabled;

/**
 * \brief Enabled serialization
 */
void serialboxEnableSerialization(void);

/**
 * \brief Disable serialization
 */
void serialboxDisableSerialization(void);

/*===------------------------------------------------------------------------------------------===*\
 *     Global Meta-information
\*===------------------------------------------------------------------------------------------===*/

/**
 * \brief Get meta-information
 *
 * The lifetime of the meta-information is tied to the lifetime of the serialboxSerializer_t object
 * and will be automatically deallocated.
 *
 * \param serializer  Serializer to use
 * \return global meta-information of the serializer
 */
serialboxMetaInfo_t serialboxSerializerGetGlobalMetaInfo(serialboxSerializer_t serializer);

/*===------------------------------------------------------------------------------------------===*\
 *     Register and Query Savepoints
\*===------------------------------------------------------------------------------------------===*/

/**
 * \brief Register savepoint ´savepoint´ within the serializer
 *
 * \param serializer  Serializer to use
 * \param savepoint   Savepoint to add
 * \return 1 if savepoint was added successfully, 0 otherwise
 */
int serialboxSerializerAddSavepoint(serialboxSerializer_t serializer,
                                    const serialboxSavepoint_t savepoint);

/**
 * \brief Get number of registered savepoints
 *
 * \param serializer  Serializer to use
 * \return Number of registered savepoints
 */
int serialboxSerializerGetNumSavepoints(const serialboxSerializer_t serializer);

/**
 * \brief Get an array of \b refrences to the registered savepoints
 *
 * The array will be allocated using malloc() and needs to be freed by the user using free(). The
 * lifetime of the savepoints however is tied to the lifetime of the ´serialboxSerializer_t´ object
 * and will thus be automatically deallocated.
 *
 * \param serializer  Serializer to use
 * \return Newly allocated array of savepoints of length ´serialboxSerializerGetNumSavepoints´
 */
serialboxSavepoint_t* serialboxSerializerGetSavepointVector(const serialboxSerializer_t serializer);

/*===------------------------------------------------------------------------------------------===*\
 *     Register and Query Fields
\*===------------------------------------------------------------------------------------------===*/

/**
 * \brief Register ´field´ within the serializer
 *
 * \param serializer  Serializer to use
 * \param name        Name of the field to register
 * \param field       Field meta-information
 * \return 1 if field was added successfully, 0 otherwise
 */
int serialboxSerializerAddField(serialboxSerializer_t serializer, const char* name,
                                const serialboxFieldMetaInfo_t fieldMetaInfo);

/**
 * \brief Get an array of C-strings of all names of the registered fields
 *
 * The function will allocate a sufficiently large array of ´char*´. Each element (as well as the
 * array itself) needs to be freed by the user using free().
 *
 * \param[in]  serializer  Serializer to use
 * \param[out] fieldnames  Array of length ´len´ of C-strings of the names of all registered fields
 * \param[out] len         Length of the array
 */
void serialboxSerializerGetFieldnames(const serialboxSerializer_t serializer, char*** fieldnames,
                                      int* len);

/**
 * \brief Get FieldMetaInfo of field with name ´name´
 *
 * The lifetime of the FieldMetaInfo is tied to the lifetime of the ´serialboxSerializer_t´ object
 * and will be automatically deallocated.
 *
 * \param serializer  Serializer to use
 * \param name        Name of the field to search for
 * \return Refrence to the FieldMetaInfo if field exists, NULL otherwise
 */
serialboxFieldMetaInfo_t serialboxSerializerGetFieldMetaInfo(const serialboxSerializer_t serializer,
                                                             const char* name);

/**
 * \brief Get an array of C-strings of the field names registered at ´savepoint´
 *
 * The function will allocate a sufficiently large array of ´char*´. Each element (as well as the
 * array itself) needs to be freed by the user using free().
 *
 * \param[in]  serializer  Serializer to use
 * \param[in]  savepoint   Savepoint of intrest
 * \param[out] fieldnames  Array of length ´len´ of C-strings of the names of all registered fields
 * \param[out] len         Length of the array
 */
void serialboxSerializerGetFieldnamesAtSavepoint(const serialboxSerializer_t serializer,
                                                 const serialboxSavepoint_t savepoint,
                                                 char*** fieldnames, int* len);

/*===------------------------------------------------------------------------------------------===*\
 *     Writing & Reading
\*===------------------------------------------------------------------------------------------===*/

/**
 * \brief Serialize field ´name´ (given by ´originPtr´ and ´strides´) at ´savepoint´ to disk
 *
 * The ´savepoint´ will be registered at field ´name´ if not yet present. The ´origingPtr´ represent
 * the memory location of the first element in the array i.e skipping all initial padding.
 *
 * \param name         Name of the field
 * \param savepoint    Savepoint to at which the field will be serialized
 * \param originPtr    Pointer to the origin of the data
 * \param strides      Array of strides of length ´numStrides´
 * \param numStrides   Number of strides
 */
void serialboxSerializerWrite(serialboxSerializer_t serializer, const char* name,
                              const serialboxSavepoint_t savepoint, void* originPtr,
                              const int* strides, int numStrides);

/**
 * \brief Deserialize field ´name´ (given by ´originPtr´ and ´strides´) at ´savepoint´ from disk
 *
 * The ´origingPtr´ represent the memory location of the first element in the array i.e skipping
 * all initial padding.
 *
 * \param name         Name of the field
 * \param savepoint    Savepoint to at which the field will be serialized
 * \param originPtr    Pointer to the origin of the data
 * \param strides      Array of strides of length ´numStrides´
 * \param numStrides   Number of strides
 */
void serialboxSerializerRead(serialboxSerializer_t serializer, const char* name,
                             const serialboxSavepoint_t savepoint, void* originPtr,
                             const int* strides, int numStrides);

#ifdef __cplusplus
}
#endif

#endif
