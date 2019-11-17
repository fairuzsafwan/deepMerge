require 'torch'
require 'nn'

local VAE = {}
local SpatialDilatedConvolution = nn.SpatialDilatedConvolution
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution
local SpatialBatchNormalization = nn.SpatialBatchNormalization


local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end


-- Residual network functions
-- The original ResNet functions were obtained from https://github.com/facebook/fb.resnet.torch and then modified

-- Typically shareGradInput uses the same gradInput storage for all modules
-- of the same type. This is incorrect for some SpatialBatchNormalization
-- modules in this network b/c of the in-place CAddTable. This marks the
-- module so that it's shared only with other modules with the same key
local function ShareGradInput(module, key)
  assert(key)
  module.__shareGradInputKey = key
  return module
end

-- The shortcut layer is either identity or 1x1 convolution
local function shortcut(nInputPlane, nOutputPlane, stride, unconv)
    local shortcutType = 'B' -- Fixed for our purposes
    local s
    local useConv = shortcutType == 'C' or
        (shortcutType == 'B' and stride and stride > 1 and nInputPlane ~= nOutputPlane)
    if useConv then
        -- Do convolution
        s = nn.Sequential()
        if not unconv or unconv == false then
            s:add(SpatialDilatedConvolution(nInputPlane, nOutputPlane, 4, 1, stride, stride, 1, 1, 2, 2))
            s:add(SpatialDilatedConvolution(nOutputPlane, nOutputPlane, 1, 4, 1, 1))
            return s            
        else
			s:add(SpatialFullConvolution(nInputPlane, nOutputPlane, 1, 4, stride, stride, 1, 1))
            s:add(SpatialFullConvolution(nOutputPlane, nOutputPlane, 4, 1, 1, 1, 0, 0))
            return s            
        end
    elseif nInputPlane ~= nOutputPlane then
        s = nn.Sequential()
        if not unconv or unconv == false then
            s:add(SpatialConvolution(nInputPlane,nOutputPlane,4,1))
            s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,4))
            return s
        else
			s:add(SpatialFullConvolution(nInputPlane,nOutputPlane,1,4))
            s:add(SpatialFullConvolution(nOutputPlane,nOutputPlane,4,1))
            return s
        end
    else
        return nn.Identity()
    end
end


local function basicblock(n, nO, stride, Type, unconv)
  local nInputPlane = n
  local nOutputPlane = nO

  local block = nn.Sequential()
  local s = nn.Sequential()

    if Type == 'both_preact' then
        block:add(ShareGradInput(SpatialBatchNormalization(nInputPlane), 'preact'))
        block:add(nn.ReLU(true))
    elseif Type ~= 'no_preact' then
        s:add(SpatialBatchNormalization(nInputPlane))
        s:add(nn.ReLU(true))
    end

    if stride and stride == 1 then
        if not unconv or unconv == false then
            s:add(SpatialConvolution(nInputPlane,nOutputPlane,3,1,1,1,1,1))
            s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,3,1,1,0,0))
        else
            s:add(SpatialFullConvolution(nInputPlane,nOutputPlane,1,3,1,1,1,1))
            s:add(SpatialFullConvolution(nOutputPlane,nOutputPlane,3,1,1,1,0,0))
        end
    elseif stride and stride > 1 then
        if not unconv or unconv == false then
            s:add(SpatialDilatedConvolution(nInputPlane, nOutputPlane, 4, 1, stride, stride, 1, 1, 2, 2))
            s:add(SpatialDilatedConvolution(nOutputPlane, nOutputPlane, 1, 4, 1, 1))
        else
            s:add(SpatialFullConvolution(nInputPlane,nOutputPlane,1,4,stride,stride,1,1))
            s:add(SpatialFullConvolution(nOutputPlane,nOutputPlane,4,1,1,1,0,0))
        end
    else
        if not unconv or unconv == false then
            s:add(SpatialConvolution(nInputPlane,nOutputPlane,4,1))
            s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,4))
        else
            s:add(SpatialFullConvolution(nInputPlane,nOutputPlane,1,4))
            s:add(SpatialFullConvolution(nOutputPlane,nOutputPlane,4,1))
        end
    end

    s:add(SpatialBatchNormalization(nOutputPlane))
    s:add(nn.ReLU(true))
    if not unconv or unconv == false then
        s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1,1,1,1,1))
        s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,3,1,1,0,0))
    else
        s:add(SpatialFullConvolution(nOutputPlane,nOutputPlane,1,3,1,1,1,1))
        s:add(SpatialFullConvolution(nOutputPlane,nOutputPlane,3,1,1,1,0,0))
    end

    local temp = nn.Sequential()
    temp:add(shortcut(nInputPlane, nOutputPlane, stride, unconv))

    return block
    :add(nn.ConcatTable()
    :add(s)
    :add(shortcut(nInputPlane, nOutputPlane, stride, unconv)))
    :add(nn.CAddTable(true))
end

--=================================== START Global Net =================================== 
local function basicblockGlobal(n, nO, stride, shortcutLevel, Type, unconv)
  local nInputPlane = n
  local nOutputPlane = nO

  local block = nn.Sequential()
  local s = nn.Sequential()

    if Type == 'both_preact' then
        block:add(ShareGradInput(SpatialBatchNormalization(nInputPlane), 'preact'))
        block:add(nn.ReLU(true))
    elseif Type ~= 'no_preact' then
        s:add(SpatialBatchNormalization(nInputPlane))
        s:add(nn.ReLU(true))
    end


    if stride and stride == 1 then
        if not unconv or unconv == false then
            s:add(SpatialConvolution(nInputPlane,nOutputPlane,1,3,2,2,0,0))
            s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1,1,1,0,0))
			s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,3,2,2,0,0))
            s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1,1,1,0,0))
        else
            s:add(SpatialFullConvolution(nInputPlane,nOutputPlane,1,3,2,2,0,0))
            s:add(SpatialFullConvolution(nOutputPlane,nOutputPlane,3,1,1,1,0,0))
			s:add(SpatialFullConvolution(nOutputPlane,nOutputPlane,1,3,2,2,0,0))
            s:add(SpatialFullConvolution(nOutputPlane,nOutputPlane,3,1,1,1,0,0))
        end
    elseif stride and stride > 1 then
        if not unconv or unconv == false then
			if Type == 'both_preact' then
				s:add(SpatialDilatedConvolution(nInputPlane, nOutputPlane, 1, 3, stride, stride, 0, 0, 2, 2))
				s:add(SpatialDilatedConvolution(nOutputPlane, nOutputPlane, 3, 1, 1, 1))
				s:add(SpatialDilatedConvolution(nOutputPlane, nOutputPlane, 1, 3, 1, 1, 0, 0, 2, 2))
				s:add(SpatialDilatedConvolution(nOutputPlane, nOutputPlane, 3, 1, 1, 1))
			else
				s:add(SpatialDilatedConvolution(nInputPlane, nOutputPlane, 1, 3, stride, stride, 0, 0, 2, 2))
				s:add(SpatialDilatedConvolution(nOutputPlane, nOutputPlane, 3, 1, 1, 1))
				s:add(SpatialDilatedConvolution(nOutputPlane, nOutputPlane, 1, 3, 1, 1, 0, 0, 2, 2))
				s:add(SpatialDilatedConvolution(nOutputPlane, nOutputPlane, 3, 1, 1, 1))
			end
        else
			s:add(SpatialDilatedConvolution(nInputPlane, nOutputPlane, 1, 5, stride, stride, 0, 0, 2, 2))
			s:add(SpatialDilatedConvolution(nOutputPlane, nOutputPlane, 5, 1, 1, 1))
        end
    else
        if not unconv or unconv == false then
			s:add(SpatialConvolution(nInputPlane,nOutputPlane,1,3))
            s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1))
			s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,3))
            s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1))
        else
			s:add(SpatialConvolution(nInputPlane,nOutputPlane,1,3))
            s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1))
			s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,3))
            s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1))
        end
    end


    s:add(SpatialBatchNormalization(nOutputPlane))
    s:add(nn.ReLU(true))
    if not unconv or unconv == false then
		if Type == 'both_preact' then
			s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,3,1,1,2,2))
			s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1,1,1,0,0))
			s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,3,1,1,1,2))
			s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1,1,1,0,0))
		else
			s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,3,1,1,2,2)) 
			s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1,1,1,0,0)) 
			s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,3,1,1,1,2)) 
			s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1,1,1,0,0))
		end
    else
        s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,3,1,1,2,2))
        s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1,1,1,0,0))
		s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,3,1,1,1,1))
        s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1,1,1,0,0))
    end

	
    return block
    :add(nn.ConcatTable()
    :add(s)
    :add(shortcut(nInputPlane, nOutputPlane, stride, unconv))) --:add(shortcut_Encoder(nInputPlane, nOutputPlane, stride, shortcutLevel, Type, unconv)))
    :add(nn.CAddTable(true))
end

local function basicblockGlobal2(n, nO, stride, shortcutLevel, Type, unconv)
  local nInputPlane = n
  local nOutputPlane = nO

  local block = nn.Sequential()
  local s = nn.Sequential()

    if Type == 'both_preact' then
        block:add(ShareGradInput(SpatialBatchNormalization(nInputPlane), 'preact'))
        block:add(nn.ReLU(true))
    elseif Type ~= 'no_preact' then
        s:add(SpatialBatchNormalization(nInputPlane))
        s:add(nn.ReLU(true))
    end

    if stride and stride == 1 then
        if not unconv or unconv == false then
            s:add(SpatialConvolution(nInputPlane,nOutputPlane,1,3,2,2,0,0))
            s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1,1,1,0,0))
			s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,3,2,2,0,0))
            s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1,1,1,0,0))
        else
            s:add(SpatialFullConvolution(nInputPlane,nOutputPlane,1,3,2,2,0,0))
            s:add(SpatialFullConvolution(nOutputPlane,nOutputPlane,3,1,1,1,0,0))
			s:add(SpatialFullConvolution(nOutputPlane,nOutputPlane,1,3,2,2,0,0))
            s:add(SpatialFullConvolution(nOutputPlane,nOutputPlane,3,1,1,1,0,0))
        end
    elseif stride and stride > 1 then
        if not unconv or unconv == false then
			if Type == 'both_preact' then
				s:add(SpatialDilatedConvolution(nInputPlane, nOutputPlane, 1, 5, stride, stride, 0, 0, 2, 2))
				s:add(SpatialDilatedConvolution(nOutputPlane, nOutputPlane, 5, 1, 1, 1))
			else
				s:add(SpatialDilatedConvolution(nInputPlane, nOutputPlane, 1, 3, stride, stride, 0, 0, 2, 2))
				s:add(SpatialDilatedConvolution(nOutputPlane, nOutputPlane, 3, 1, 1, 1))
				s:add(SpatialDilatedConvolution(nOutputPlane, nOutputPlane, 1, 3, 1, 1, 0, 0, 2, 2))
				s:add(SpatialDilatedConvolution(nOutputPlane, nOutputPlane, 3, 1, 1, 1))
			end
        else
			s:add(SpatialDilatedConvolution(nInputPlane, nOutputPlane, 1, 5, stride, stride, 0, 0, 2, 2))
			s:add(SpatialDilatedConvolution(nOutputPlane, nOutputPlane, 5, 1, 1, 1))
        end
    else
        if not unconv or unconv == false then
			s:add(SpatialConvolution(nInputPlane,nOutputPlane,1,3))
            s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1))
			s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,3))
            s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1))
        else
			s:add(SpatialConvolution(nInputPlane,nOutputPlane,1,3))
            s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1))
			s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,3))
            s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1))
        end
    end


    s:add(SpatialBatchNormalization(nOutputPlane))
    s:add(nn.ReLU(true))
    if not unconv or unconv == false then
		if Type == 'both_preact' then
			s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,5,1,1,3,3))
			s:add(SpatialConvolution(nOutputPlane,nOutputPlane,5,1,1,1,0,0))
		else
			s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,3,1,1,2,2))
			s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1,1,1,0,0)) 
			s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,3,1,1,1,2)) 
			s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1,1,1,0,0)) 
		end
    else
        s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,3,1,1,2,2))
        s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1,1,1,0,0))
		s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,3,1,1,1,1))
        s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1,1,1,0,0))
    end
	
    return block
    :add(nn.ConcatTable()
    :add(s)
    :add(shortcut(nInputPlane, nOutputPlane, stride, unconv))) --:add(shortcut_Encoder(nInputPlane, nOutputPlane, stride, shortcutLevel, Type, unconv)))
    :add(nn.CAddTable(true))
end
--=================================== END Global Net =================================== 

--=================================== START Low Net =================================== 
local function basicblockLow(n, nO, stride, shortcutLevel, Type, unconv)
    local nInputPlane = n
    local nOutputPlane = nO

    local block = nn.Sequential()
    local s = nn.Sequential()

    if Type == 'both_preact' then
        block:add(ShareGradInput(SpatialBatchNormalization(nInputPlane), 'preact'))
        block:add(nn.ReLU(true))
    elseif Type ~= 'no_preact' then
        s:add(SpatialBatchNormalization(nInputPlane))
        s:add(nn.ReLU(true))
    end


    if stride and stride == 1 then
        if not unconv or unconv == false then
            s:add(SpatialConvolution(nInputPlane,nOutputPlane,1,1,2,2,0,0))
            s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,1,1,1,0,0))
        else
            s:add(SpatialFullConvolution(nInputPlane,nOutputPlane,1,1,2,2,0,0))
            s:add(SpatialFullConvolution(nOutputPlane,nOutputPlane,1,1,1,1,0,0))
        end
    elseif stride and stride > 1 then
        if not unconv or unconv == false then
            s:add(SpatialDilatedConvolution(nInputPlane, nOutputPlane, 1, 2, stride, stride, 0, 0, 2, 2))
            s:add(SpatialDilatedConvolution(nOutputPlane, nOutputPlane, 2, 1, 1, 1))
        else
			s:add(SpatialDilatedConvolution(nInputPlane, nOutputPlane, 1, 2, stride, stride, 0, 0, 2, 2))
            s:add(SpatialDilatedConvolution(nOutputPlane, nOutputPlane, 2, 1, 1, 1))
        end
    else
        if not unconv or unconv == false then
            s:add(SpatialConvolution(nInputPlane,nOutputPlane,1,1))
            s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,1))
        else
            s:add(SpatialFullConvolution(nInputPlane,nOutputPlane,1,1))
            s:add(SpatialFullConvolution(nOutputPlane,nOutputPlane,1,1))
        end
    end


    s:add(SpatialBatchNormalization(nOutputPlane))
    s:add(nn.ReLU(true))
    if not unconv or unconv == false then
        s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,2,1,1,0,0))
        s:add(SpatialConvolution(nOutputPlane,nOutputPlane,2,1,1,1,0,0))
    else
		s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,2,2,2,1,1))
		s:add(SpatialConvolution(nOutputPlane,nOutputPlane,2,1,2,2,1,1))
    end

    return block
    :add(nn.ConcatTable()
    :add(s)
    :add(shortcut(nInputPlane, nOutputPlane, stride, unconv))) --:add(shortcut_Encoder(nInputPlane, nOutputPlane, stride, shortcutLevel, Type, unconv)))
    :add(nn.CAddTable(true))
end
--=================================== END Low Net =================================== 

--=================================== START Mid Net =================================== 
local function basicblockMid(n, nO, stride, shortcutLevel, Type, unconv)
    local nInputPlane = n
    local nOutputPlane = nO

    local block = nn.Sequential()
    local s = nn.Sequential()

    if Type == 'both_preact' then
        block:add(ShareGradInput(SpatialBatchNormalization(nInputPlane), 'preact'))
        block:add(nn.ReLU(true))
    elseif Type ~= 'no_preact' then
        s:add(SpatialBatchNormalization(nInputPlane))
        s:add(nn.ReLU(true))
    end


    if stride and stride == 1 then
        if not unconv or unconv == false then
            s:add(SpatialConvolution(nInputPlane,nOutputPlane,1,3,2,2,0,0))
            s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1,1,1,0,0))
        else
            s:add(SpatialFullConvolution(nInputPlane,nOutputPlane,1,3,2,2,0,0))
            s:add(SpatialFullConvolution(nOutputPlane,nOutputPlane,3,1,1,1,0,0))
        end
    elseif stride and stride > 1 then
        if not unconv or unconv == false then
            s:add(SpatialDilatedConvolution(nInputPlane, nOutputPlane, 1, 3, stride, stride, 0, 0, 2, 2))
            s:add(SpatialDilatedConvolution(nOutputPlane, nOutputPlane, 3, 1, 1, 1))
        else
			s:add(SpatialDilatedConvolution(nInputPlane, nOutputPlane, 1, 3, stride, stride, 0, 0, 2, 2))
            s:add(SpatialDilatedConvolution(nOutputPlane, nOutputPlane, 3, 1, 1, 1))
        end
    else
        if not unconv or unconv == false then
            s:add(SpatialConvolution(nInputPlane,nOutputPlane,1,3))
            s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1))
        else
            s:add(SpatialFullConvolution(nInputPlane,nOutputPlane,1,3))
            s:add(SpatialFullConvolution(nOutputPlane,nOutputPlane,3,1))
        end
    end


    s:add(SpatialBatchNormalization(nOutputPlane))
    s:add(nn.ReLU(true))
    if not unconv or unconv == false then
		s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,3,1,1,1,1))
        s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1,1,1,0,0))
    else
        s:add(SpatialConvolution(nOutputPlane,nOutputPlane,1,3,2,2,0,0))
        s:add(SpatialConvolution(nOutputPlane,nOutputPlane,3,1,1,1,0,0))
    end

    return block
    :add(nn.ConcatTable()
    :add(s)
    :add(shortcut(nInputPlane, nOutputPlane, stride, unconv))) --:add(shortcut_Encoder(nInputPlane, nOutputPlane, stride, shortcutLevel, Type, unconv)))
    :add(nn.CAddTable(true))
end
--=================================== END Mid Net =================================== 

local function residualBlock(block, nInputPlane, nOutputPlane, count, stride, Type, unconv)
  local s = nn.Sequential()
  if count < 1 then
    return s
  end
  s:add(block(nInputPlane, nOutputPlane, stride,
              Type == 'first' and 'no_preact' or 'both_preact', unconv))
  for i=2,count do
     s:add(block(nOutputPlane, nOutputPlane, 1))
  end
  return s
end


local function residualBlock_Encoder(block, nInputPlane, nOutputPlane, count, stride, shortcutLevel, Type, unconv)
  local s = nn.Sequential()
  if count < 1 then
    return s
  end
  s:add(block(nInputPlane, nOutputPlane, stride, shortcutLevel,
              Type == 'first' and 'no_preact' or 'both_preact', unconv))
  for i=2,count do
     s:add(block(nOutputPlane, nOutputPlane, 1, shortcutLevel))
  end
  return s
end


function VAE.get_encoder(modelParams)

    local nInputCh = modelParams[1] -- or number of view points ie. 20
    local nOutputCh = modelParams[2] -- 74
    local nLatents = modelParams[3] -- 100
    local singleVPNet = modelParams[5] -- 0
    local conditional = modelParams[6] -- 0
    local numCats = modelParams[7] -- 0
    local benchmark = modelParams[8] -- 0
    local dropoutNet = modelParams[9] -- 0

	local base = nn.Sequential()
	base:add(SpatialDilatedConvolution(not singleVPNet and nInputCh or 1, nOutputCh * 4, 4, 4, 2, 2, 1, 1, 2, 2))
    base:add(SpatialBatchNormalization(nOutputCh * 4)):add(nn.ReLU(true))
    base:forward(torch.zeros(4, 20, 224, 224))
	
	local globalNet = nn.Sequential()
	globalNet:add(base:clone('weight', 'bias', 'gradWeight', 'gradBias', 'running_mean', 'runnig_var', 'save_mean', 'save_var'))
	globalNet:add(residualBlock_Encoder(basicblockGlobal, nOutputCh * 4, nOutputCh * 6,  1, 2, 'global', 'first', false))
    -- Output feature map size: 53 x 53
	globalNet:add(residualBlock_Encoder(basicblockGlobal, nOutputCh * 6, nOutputCh * 8,  1, 2, 'global', nil, false))
	-- Output feature map size: 25 x 25
	globalNet:add(residualBlock_Encoder(basicblockGlobal, nOutputCh * 8, nOutputCh * 6,  1, 2, 'global', nil, false))
	-- Output feature map size: 11 x 11
	globalNet:add(residualBlock_Encoder(basicblockGlobal2, nOutputCh * 6, nOutputCh * 1,  1, 2, 'global', nil, false))
	-- Output feature map size: 4 x 4
	
	
	local midNet = nn.Sequential()
	midNet:add(base:clone('weight', 'bias', 'gradWeight', 'gradBias', 'running_mean', 'runnig_var', 'save_mean', 'save_var'))
	midNet:add(residualBlock_Encoder(basicblockMid, nOutputCh * 4, nOutputCh * 6,  1, 2, 'mid', 'first', false))
	 -- Output feature map size: 53 x 53
	midNet:add(residualBlock_Encoder(basicblockMid, nOutputCh * 6, nOutputCh * 8, 1, 2, 'mid', nil, false)) 
    -- Output feature map size: 25 x 25
	midNet:add(residualBlock_Encoder(basicblockMid, nOutputCh * 8, nOutputCh * 6, 1, 2, 'mid', nil, false))
    -- Output feature map size: 11 x 11
	midNet:add(residualBlock_Encoder(basicblockMid, nOutputCh * 6, nOutputCh * 1, 1, 2, 'mid', nil, false))
    -- Output feature map size: 4 x 4
   
	local lowNet = nn.Sequential()
	lowNet:add(base:clone('weight', 'bias', 'gradWeight', 'gradBias', 'running_mean', 'runnig_var', 'save_mean', 'save_var'))
	lowNet:add(residualBlock_Encoder(basicblockLow, nOutputCh * 4, nOutputCh * 6,  1, 2, 'low', 'first', false))
	 -- Output feature map size: 53 x 53
	lowNet:add(residualBlock_Encoder(basicblockLow, nOutputCh * 6, nOutputCh * 8, 1, 2, 'low', nil, false))
    -- Output feature map size: 25 x 25
	lowNet:add(residualBlock_Encoder(basicblockLow, nOutputCh * 8, nOutputCh * 6, 1, 2, 'low', nil, false))
    -- Output feature map size: 11 x 11
	lowNet:add(residualBlock_Encoder(basicblockLow, nOutputCh * 6, nOutputCh * 1, 1, 2, 'low', nil, false))
    -- Output feature map size: 4 x 4

	local mergeNet = nn.Concat(2)
	mergeNet:add(globalNet)
	mergeNet:add(midNet)
    mergeNet:add(lowNet)
	
	local finalEncoder = nn.Sequential()
	finalEncoder:add(mergeNet)
	finalEncoder:add(nn.SpatialBatchNormalization(nOutputCh*3))
	finalEncoder:add(nn.View(nOutputCh * 3 * 4 * 4):setNumInputDims(3))
    
    local mean_logvar = nn.ConcatTable()
    if not benchmark or singleVPNet or dropoutNet then
        mean_logvar:add(nn.Sequential():add(nn.Linear(nOutputCh * 3 * 4 * 4, nOutputCh * 4 * 2)):add(nn.ReLU(true)):add(nn.Linear(nOutputCh * 4 * 2, nLatents))) -- The means
        mean_logvar:add(nn.Sequential():add(nn.Linear(nOutputCh * 3 * 4 * 4, nOutputCh * 4 * 2)):add(nn.ReLU(true)):add(nn.Linear(nOutputCh * 4 * 2, nLatents))) -- Log of the variances
    else
        mean_logvar:add(nn.Linear(nOutputCh * 4 * 4, nLatents)) -- The means
        mean_logvar:add(nn.Linear(nOutputCh * 4 * 4, nLatents)) -- Log of the variances
    end
    if conditional then
        mean_logvar:add(nn.Sequential()
        :add(nn.Linear(nOutputCh * 4 * 4, (nOutputCh * 4 * 4) - 50))
        :add(nn.ReLU(true))
        :add(nn.Linear((nOutputCh * 4 * 4) - 50, numCats)))
    end

	finalEncoder:add(mean_logvar)
	finalEncoder:apply(weights_init)
    finalEncoder:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

	return finalEncoder
	--
end

function VAE.get_decoder(modelParams)

    local nInputCh = modelParams[1]
    local nOutputCh = modelParams[2]
    local nLatents = modelParams[3]
    local tanh = modelParams[4]
    local singleVPNet = modelParams[5]
    local conditional = modelParams[6]
    local numCats = modelParams[7]
    local benchmark = modelParams[8]
    local dropoutNet = modelParams[9]

    local decoder = nn.Sequential()
    if conditional then
        decoder:add(nn.JoinTable(2))
    end
    
    if not benchmark or singleVPNet or dropoutNet then
        decoder:add(nn.Linear(nLatents+numCats, nOutputCh * 4 * 4)):add(nn.ReLU(true))
        decoder:add(nn.Linear(nOutputCh * 4 * 4, nOutputCh * 2 * 4 * 4))
    else
        decoder:add(nn.Linear(nLatents+numCats, nOutputCh * 2 * 4 * 4))
    end
    decoder:add(nn.View(nOutputCh * 2 , 4, 4))
    decoder:add(SpatialBatchNormalization(nOutputCh * 2)):add(nn.ReLU(true))
    -- Output feature map size: 4 x 4
    decoder:add(residualBlock(basicblock, nOutputCh * 2, nOutputCh * 6,  1, nil, 'first', true))
    -- Output feature map size: 7 x 7
    decoder:add(residualBlock(basicblock, nOutputCh * 6, nOutputCh * 8,  1, 2, nil, true))
    -- Output feature map size: 14 x 14
    decoder:add(residualBlock(basicblock, nOutputCh * 8, nOutputCh * 7,  1, 2, nil, true))
    decoder:add(ShareGradInput(SpatialBatchNormalization(nOutputCh * 7), 'last'))
    -- Output feature map size: 28 x 28

    decoder:add(SpatialFullConvolution(nOutputCh * 7, nOutputCh * 6, 4, 4, 2, 2, 1, 1))
    decoder:add(SpatialBatchNormalization(nOutputCh * 6)):add(nn.ReLU(true))
    -- Output feature map size: 56 x 56

    decoder:add(SpatialFullConvolution(nOutputCh * 6, nOutputCh * 4, 4, 4, 2, 2, 1, 1))
    decoder:add(SpatialBatchNormalization(nOutputCh * 4)):add(nn.ReLU(true))
    -- Output feature map size: 112 x 112

    -- temoDeconvLayer1 generates the depth maps
    tempDeconvLayer1 = nn.Sequential():add(SpatialFullConvolution(nOutputCh * 4, nInputCh, 4, 4, 2, 2, 1, 1))
    if tanh then
        tempDeconvLayer1:add(nn.Tanh())
    else
        tempDeconvLayer1:add(nn.Sigmoid())
    end
    -- temoDeconvLayer2 generates the silhouettes
    tempDeconvLayer2 = nn.Sequential():add(SpatialFullConvolution(nOutputCh * 4, nInputCh, 4, 4, 2, 2, 1, 1)):add(nn.Sigmoid())
    decoder:add(nn.ConcatTable():add(tempDeconvLayer1):add(tempDeconvLayer2))
    -- Output feature map size: 224 x 224

    decoder:apply(weights_init)
    decoder:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

    return decoder
end

return VAE
