import numpy as np
# import cfdtool.IO as io
import cfdtool.Math as mth
import cfdtool.Interpolate as interp


def cfdRhieChowValue(fieldName,Region):
    numberOfInteriorFaces = Region.mesh.numberOfInteriorFaces
    ## owners of elements of faces
    owners_f=Region.mesh.interiorFaceOwners
    ## neighbour elements of faces
    neighbours_f=Region.mesh.interiorFaceNeighbours
    ## face weights
    g_f=Region.mesh.interiorFaceWeights
    ## vector formed between owner (C) and neighbour (f) elements
    CF = Region.mesh.interiorFaceCF

    ## face gradient matrix
    gradPhi=np.squeeze(Region.fluid[fieldName].Grad.phi[:numberOfInteriorFaces,:])
    grad_f=(1-g_f)[:,None]*gradPhi[neighbours_f,:]+g_f[:,None]*gradPhi[owners_f,:]
    # % ScfdUrface-normal gradient
    dcfdMag = mth.cfdMag(CF)
    e_CF = mth.cfdUnit(CF.value)
    # local_avg_grad=(grad_f*e_CF).sum(1)*e_CF
    # % Get pressure field and interpolate to faces
    phi = cfdGetSubArrayForInterior(fieldName,Region)
    local_grad_cfdMag_f = np.squeeze((phi[neighbours_f]-phi[owners_f]))/dcfdMag
    # 《the FVM in CFD》 P.289
    return (local_grad_cfdMag_f-mth.cfdDot(grad_f,e_CF))[:,None]*e_CF
    # return  RhieChow_grad

def cfdGetfield_grad_f(fieldName,Region):
    # % Get pressure field and interpolate to faces
    field_Interior = cfdGetSubArrayForInterior(fieldName,Region)
    Region.fluid[fieldName].Grad.cfdGetGradientSubArrayForInterior(Region)
    return  interp.cfdInterpolateGradientsFromElementsToInteriorFaces(Region,Region.fluid[fieldName].Grad.phiInter, 'Gauss linear corrected', field_Interior)
    # field_grad_f = 
    # return  field_grad_f

def cfdGetSubArrayForInterior(theFieldName,Region,*args):
    Fieldtype=Region.fluid[theFieldName].type
    if Fieldtype == 'surfaceScalarField':
        return Region.fluid[theFieldName].phi[:Region.mesh.numberOfInteriorFaces]
    elif Fieldtype == 'surfaceVectorField':
        if args:
            iComponent = args[0]
            return  Region.fluid[theFieldName].phi[:Region.mesh.numberOfInteriorFaces,iComponent] 
        else:
            phiInteriorSubArray =  Region.fluid[theFieldName].phi[:Region.mesh.numberOfInteriorFaces, :]
    elif Fieldtype == 'volScalarField':
        return  Region.fluid[theFieldName].phi[:Region.mesh.numberOfElements]    
    elif Fieldtype == 'volVectorField':
        if args:
            iComponent = args[0]
            return  Region.fluid[theFieldName].phi[:Region.mesh.numberOfElements,iComponent] 
        else:
            return  Region.fluid[theFieldName].phi[:Region.mesh.numberOfElements, :]
    else:
        raise ValueError('Field type not supported')



def cfdGetSubArrayForBoundaryPatch(theFieldName, iBPatch, Region,*args):
# %==========================================================================
# % Routine Description:
# %   This function returns a subarray at a given cfdBoundary
# %--------------------------------------------------------------------------
    if Region.fluid[theFieldName].type=='surfaceScalarField':
        # iFaceStart = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex']
        # iFaceEnd = iFaceStart+Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces']
        iBFaces = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBFaces']
        phi_b = Region.fluid[theFieldName].phi[iBFaces]
    elif Region.fluid[theFieldName].type=='volScalarField':
        iBElements=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBElements']
        phi_b = Region.fluid[theFieldName].phi[iBElements]
    
    elif Region.fluid[theFieldName].type=='volVectorField':
        iBElements=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBElements']
        if args:
            iComponent=args[0]
            phi_b = Region.fluid[theFieldName].phi[iBElements,iComponent]
        else:
            phi_b = Region.fluid[theFieldName].phi[iBElements,:]
    return np.squeeze(phi_b)

def cfdGetGradientSubArrayForBoundaryPatch(theFieldName, iBPatch, Region,*args):
# %==========================================================================
# % Routine Description:
# %   This function returns the gradient subarray at a given cfdBoundary
# %--------------------------------------------------------------------------
    if Region.fluid[theFieldName].type=='scfdUrfaceScalarField':
        # iFaceStart = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex']
        # iFaceEnd = iFaceStart+Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces']
        iBFaces = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBFaces']
        phiGrad_b = Region.fluid[theFieldName].Grad.phi[iBFaces, :]
    elif Region.fluid[theFieldName].type=='volScalarField':
        # iElementStart = Region.mesh.numberOfElements+Region.mesh.cfdBoundaryPatchesArray{iBPatch}.startFaceIndex-Region.mesh.numberOfInteriorFaces;
        # iElementEnd = iElementStart+Region.mesh.cfdBoundaryPatchesArray{iBPatch}.numberOfBFaces-1;
        # iBElements = iElementStart:iElementEnd;
        iBElements=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBElements']
        phiGrad_b = Region.fluid[theFieldName].Grad.phi[iBElements, :]
    
    elif Region.fluid[theFieldName].type=='volVectorField':
        iBElements=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBElements']
        if args:
            iComponent=args[0]
            phiGrad_b = Region.fluid[theFieldName].Grad.phi[iBElements, :, iComponent]
        else:
            phiGrad_b = Region.fluid[theFieldName].Grad.phi[iBElements, :, :]

    return np.squeeze(phiGrad_b)