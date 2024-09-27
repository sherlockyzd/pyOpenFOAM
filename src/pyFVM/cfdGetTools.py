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
    ## vector of ones
    ones=np.ones((numberOfInteriorFaces))

    ## face gradient matrix
    gradPhi=Region.fluid[fieldName].phiGrad.cfdGetGradientSubArrayForInterior(Region)
    # Region.fluid[fieldName].phiGrad.phiGradInter
    grad_f=(ones-g_f)[:,None]*gradPhi[neighbours_f,:]+np.asarray(g_f)[:,None]*gradPhi[owners_f,:]
    # % ScfdUrface-normal gradient
    dcfdMag = mth.cfdMag(CF)
    e_CF = mth.cfdUnit(CF)
    # local_avg_grad=(grad_f*e_CF).sum(1)*e_CF
    # % Get pressure field and interpolate to faces
    phi = cfdGetSubArrayForInterior(fieldName,Region)
    local_grad_cfdMag_f = np.squeeze((phi[neighbours_f]-phi[owners_f]))/dcfdMag
    RhieChow_grad=(local_grad_cfdMag_f-mth.cfdDot(grad_f,e_CF))[:,None]*e_CF

    return  RhieChow_grad

def cfdGetfield_grad_f(fieldName,Region):
    # % Get pressure field and interpolate to faces
    field_Interior = cfdGetSubArrayForInterior(fieldName,Region)
    Region.fluid[fieldName].phiGrad.cfdGetGradientSubArrayForInterior(Region)
    field_grad_f =  interp.cfdInterpolateGradientsFromElementsToInteriorFaces(Region,Region.fluid[fieldName].phiGrad.phiGradInter, 'Gauss linear corrected', field_Interior)
    return  field_grad_f

def cfdGetSubArrayForInterior(theFieldName,Region,*args):
    Fieldtype=Region.fluid[theFieldName].type
    phi=Region.fluid[theFieldName].phi
    if Fieldtype == 'surfaceScalarField':
        phiInteriorSubArray =  phi[0:Region.mesh.numberOfInteriorFaces]
    elif Fieldtype == 'volScalarField':
        phiInteriorSubArray =  phi[0:Region.mesh.numberOfElements]    
    elif Fieldtype == 'volVectorField':
        if args:
            iComponent = args[0]
            phiInteriorSubArray =  phi[0:Region.mesh.numberOfElements,iComponent] 
        else:
            phiInteriorSubArray =  phi[0:Region.mesh.numberOfElements, :]
    return phiInteriorSubArray


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
        phiGrad_b = Region.fluid[theFieldName].phiGrad.phiGrad[iBFaces, :]
    elif Region.fluid[theFieldName].type=='volScalarField':
        # iElementStart = Region.mesh.numberOfElements+Region.mesh.cfdBoundaryPatchesArray{iBPatch}.startFaceIndex-Region.mesh.numberOfInteriorFaces;
        # iElementEnd = iElementStart+Region.mesh.cfdBoundaryPatchesArray{iBPatch}.numberOfBFaces-1;
        # iBElements = iElementStart:iElementEnd;
        iBElements=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBElements']
        phiGrad_b = Region.fluid[theFieldName].phiGrad.phiGrad[iBElements, :]
    
    elif Region.fluid[theFieldName].type=='volVectorField':
        iBElements=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBElements']
        if args:
            iComponent=args[0]
            phiGrad_b = Region.fluid[theFieldName].phiGrad.phiGrad[iBElements, :, iComponent]
        else:
            phiGrad_b = Region.fluid[theFieldName].phiGrad.phiGrad[iBElements, :, :]

    return np.squeeze(phiGrad_b)